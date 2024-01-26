# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
from typing import List

import os
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer
from clip_encoder import Tokenizer, CLIPVisionTower, create_request_vision_tower, create_request_noimg, create_request_feat


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        model_config = json.loads(args['model_config'])
		
        params = model_config["parameters"]
        self.schema = params["schema"]["string_value"]
        self.hidden_size = int(params["hidden_size"]["string_value"])
        self.max_input_length = int(params["max_input_len"]["string_value"])
        self.model_name = params["model_name"]["string_value"]
		
        tokenizer_dir = model_config['parameters']['tokenizer_dir'][
            'string_value']
        tokenizer_type = model_config['parameters']['tokenizer_type'][
            'string_value']
        self.add_special_tokens = model_config['parameters'].get(
            'add_special_tokens',
            {'string_value': "false"})['string_value'].lower() in [
                'true', '1', 't', 'y', 'yes'
            ]

        if 'llava' in self.model_name.lower():
            # llava start
            self.llava = tokenizer_dir
            self.vt = None # vision tower
            if self.schema == "vision_tower":
                sdevice = f"cuda:{args.get('model_instance_device_id', '0')}"
                self.vt = CLIPVisionTower(self.llava, sdevice)
            self.tk = Tokenizer(self.llava)
            self.tokenizer = self.tk.tokenizer
        elif tokenizer_type == 't5':
            self.tokenizer = T5Tokenizer(vocab_file=tokenizer_dir,
                                         padding_side='left')
        elif tokenizer_type == 'chatglm':
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                           trust_remote_code=True)
        elif tokenizer_type == 'baichuan':
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                           use_fast=False,
                                                           trust_remote_code=True)
        elif tokenizer_type == 'auto':
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, padding_side='left', trust_remote_code=True)
        elif tokenizer_type == 'llama':
            self.tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_dir, legacy=False, padding_side='left')
        else:
            raise AttributeError(
                f'Unexpected tokenizer type: {tokenizer_type}')

        if tokenizer_type == 'chatglm':
            self.tokenizer_pad_id = self.tokenizer.pad_token_id
            self.tokenizer_end_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer_end_id = self.tokenizer.encode(
                self.tokenizer.eos_token, add_special_tokens=False)[0]
            self.tokenizer_pad_id = self.tokenizer.encode(
                self.tokenizer.pad_token, add_special_tokens=False)[0]

        # Parse model output configs and convert Triton types to numpy types
        output_names = [
            "INPUT_ID", "REQUEST_INPUT_LEN", "BAD_WORDS_IDS", "STOP_WORDS_IDS",
            "OUT_END_ID", "OUT_PAD_ID",
            "IMAGE_FEATURE",
            "FEATURE_PATH"
        ]
        input_names = ["EMBEDDING_BIAS_WORDS", "EMBEDDING_BIAS_WEIGHTS"]
        for input_name in input_names:
            setattr(
                self,
                input_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_input_config_by_name(
                        model_config, input_name)['data_type']))

        for output_name in output_names:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, output_name)['data_type']))


    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        logger = pb_utils.Logger
        for idx, request in enumerate(requests):
            # Get input tensors
            query = pb_utils.get_input_tensor_by_name(request,
                                                      'QUERY').as_numpy()
            batch_dim = query.shape[0]
            if batch_dim != 1:

                err_str = "Inflight batching backend expects requests with batch size of 1."
                logger.log_error(err_str)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(err_str)))
                continue

            request_output_len = pb_utils.get_input_tensor_by_name(
                request, 'REQUEST_OUTPUT_LEN').as_numpy()

            # multimodel processing
            images = None
            feat_size = None
            
            # llava
            if 'llava' in self.model_name.lower():
                images = pb_utils.get_input_tensor_by_name(request, "IMAGES")
                feat_size = pb_utils.get_input_tensor_by_name(request, "FEATURE_SIZE")

            if images is not None:
                images = images.as_numpy()
            if feat_size is not None:
                feat_size = feat_size.as_numpy()

            bad_words_dict = pb_utils.get_input_tensor_by_name(
                request, 'BAD_WORDS_DICT')
            if bad_words_dict is not None:
                bad_words_dict = bad_words_dict.as_numpy()

            stop_words_dict = pb_utils.get_input_tensor_by_name(
                request, 'STOP_WORDS_DICT')
            if stop_words_dict is not None:
                stop_words_dict = stop_words_dict.as_numpy()

            embedding_bias_words = pb_utils.get_input_tensor_by_name(
                request, 'EMBEDDING_BIAS_WORDS')
            if embedding_bias_words is not None:
                embedding_bias_words = embedding_bias_words.as_numpy()

            embedding_bias_weights = pb_utils.get_input_tensor_by_name(
                request, 'EMBEDDING_BIAS_WEIGHTS')
            if embedding_bias_weights is not None:
                embedding_bias_weights = embedding_bias_weights.as_numpy()

            # Take the end_id from the input tensors
            # If not specified, use tokenizer to get end_id
            end_id = pb_utils.get_input_tensor_by_name(request, 'END_ID')
            if end_id is not None:
                end_id = end_id.as_numpy()
            else:
                end_id = [[self.tokenizer_end_id]]

            # Take the pad_id from the input tensors
            # If not specified, use tokenizer to get pad_id
            pad_id = pb_utils.get_input_tensor_by_name(request, 'PAD_ID')
            if pad_id is not None:
                pad_id = pad_id.as_numpy()
            else:
                pad_id = [[self.tokenizer_pad_id]]

            # Preprocessing input data.
            feats = None  # deliver feature data to next model
            feat_path = (
                None  # save feature to a file and deliver file path to next model
            )
            if "llava" in self.model_name.lower():
                # llava
                if images is not None:
                    if self.schema == "input_feature":
                        # input feature requires feature size and feature file path,
                        # feature size is used to insert placeholders into input-ids
                        # feature file will be loaded inside trtllm backend and saved
                        # for lookup-plugin
                        assert feat_size is not None
                        input_id, request_input_len, feat_path = create_request_feat(
                            self.tk,
                            query,
                            images,
                            feat_size[0][0],
                            self.tokenizer_pad_id,
                            self.hidden_size,
                        )
                    elif self.schema == "vision_tower":
                        assert self.vt is not None
                        (
                            input_id,
                            request_input_len,
                            feats,
                        ) = create_request_vision_tower(
                            self.tk, self.vt, query, images, self.tokenizer_pad_id
                        )
                        feats = np.array(feats, dtype=self.image_feature_dtype)
                    else:
                        raise Exception(f"unknown schema {self.schema}")
                else:
                    input_id, request_input_len = create_request_noimg(
                        self.tk, query, self.tokenizer_pad_id
                    )
            else:
                input_id, request_input_len = self._create_request(query)

            # jgq: check input size, request will fail if it exceeds max input length
            if any(request_input_len > self.max_input_length):
                err_str = f"input token size {request_input_len} exceeds max input length {self.max_input_length}"
                logger.log_error(err_str)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(err_str)
                    )
                )
                continue
            bad_words = self._to_word_list_format(bad_words_dict)
            stop_words = self._to_word_list_format(stop_words_dict)

            embedding_bias = self._get_embedding_bias(
                embedding_bias_words, embedding_bias_weights,
                self.embedding_bias_weights_dtype)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            input_id_tensor = pb_utils.Tensor(
                'INPUT_ID', input_id.astype(self.input_id_dtype))
            request_input_len_tensor = pb_utils.Tensor(
                'REQUEST_INPUT_LEN',
                request_input_len.astype(self.request_input_len_dtype))
            request_output_len_tensor = pb_utils.Tensor(
                'REQUEST_OUTPUT_LEN', request_output_len)
            bad_words_ids_tensor = pb_utils.Tensor('BAD_WORDS_IDS', bad_words)
            stop_words_ids_tensor = pb_utils.Tensor('STOP_WORDS_IDS',
                                                    stop_words)
            embedding_bias_tensor = pb_utils.Tensor('EMBEDDING_BIAS',
                                                    embedding_bias)
            end_id_tensor = pb_utils.Tensor('OUT_END_ID',
                                            np.array(end_id, dtype=np.int32))
            pad_id_tensor = pb_utils.Tensor('OUT_PAD_ID',
                                            np.array(pad_id, dtype=np.int32))

			# llava
            if feats is None:
                feats = np.zeros((input_id.shape[0], 1, 1, 1)).astype(
                    self.image_feature_dtype
                )
            if feat_path is None:
                feat_path = np.empty((input_id.shape[0], 1), self.feature_path_dtype)
            image_feat_tensor = pb_utils.Tensor( "IMAGE_FEATURE", feats)
            feature_path_tensor = pb_utils.Tensor( "FEATURE_PATH", feat_path)
			
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_id_tensor, bad_words_ids_tensor, stop_words_ids_tensor,
                request_input_len_tensor, request_output_len_tensor,
                embedding_bias_tensor, 
				end_id_tensor, pad_id_tensor,
                feature_path_tensor,
                image_feat_tensor,
            ])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    def _create_request(self, query):
        """
            query : batch string (2D numpy array)
        """
        start_ids = [
            np.array(
                self.tokenizer.encode(
                    s[0].decode(),
                    add_special_tokens=self.add_special_tokens)).astype(int)
            for s in query
        ]
        start_lengths = np.array([[len(ids)] for ids in start_ids]).astype(int)

        max_len = 0
        for seq in start_ids:
            max_len = max(max_len, seq.shape[0])
        start_ids = np.stack([
            np.pad(seq, (0, max_len - seq.shape[0]),
                   'constant',
                   constant_values=(0, self.tokenizer_pad_id))
            for seq in start_ids
        ])

        return start_ids, start_lengths

    def _to_word_list_format(self, word_lists: List[List[str | bytes]]):
        '''
        word_lists format:
            len(word_lists) == batch_size
            word_lists[i] means the words associated to batch item i. A "word" may actually be any string. Like "lorem" or "lorem ipsum".
        '''
        assert self.tokenizer != None, "need to set tokenizer"

        if word_lists is None:
            # Return an empty array of shape (1,2,0)
            return np.empty([1, 2, 0], dtype="int32")

        flat_ids = []
        offsets = []
        for word_list in word_lists:
            item_flat_ids = []
            item_offsets = []

            for word in word_list:
                if isinstance(word, bytes):
                    word = word.decode()

                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)),
                                 constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)),
                                constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose(
            (1, 0, 2))

    def _get_embedding_bias(self, embedding_bias_words, embedding_bias_weights,
                            bias_dtype):

        assert self.tokenizer != None, "need to set tokenizer"

        if embedding_bias_words is None or embedding_bias_weights is None:
            return np.empty([1, 0], dtype=self.embedding_bias_weights_dtype)

        batch_embedding_bias = []
        for words, weights in zip(embedding_bias_words,
                                  embedding_bias_weights):

            vocab_size = self.tokenizer.vocab_size
            embedding_bias = [0.] * vocab_size

            assert len(words) == len(
                weights
            ), "Embedding bias words must have same dimension as embedding bias weights"

            for word, weight in zip(words, weights):
                if isinstance(word, bytes):
                    word = word.decode()
                ids = self.tokenizer.encode(word)

                if len(ids) == 0:
                    continue

                for id in ids:
                    embedding_bias[id] += weight

            batch_embedding_bias.append(np.array(embedding_bias))

        return np.array(batch_embedding_bias, dtype=bias_dtype)

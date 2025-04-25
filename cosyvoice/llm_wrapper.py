import onnxruntime
import numpy as np
import torch



class Qwen2LMOnnxWrapper:
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    @torch.inference_mode()
    def inference(self,
                  text: torch.Tensor,
                  text_len: torch.Tensor,
                  prompt_text: torch.Tensor,
                  prompt_text_len: torch.Tensor,
                  prompt_speech_token: torch.Tensor,
                  prompt_speech_token_len: torch.Tensor,
                  embedding: torch.Tensor,
                  sampling: int = 25,
                  max_token_text_ratio: float = 20,
                  min_token_text_ratio: float = 2):

        inputs = {
            "text": text.cpu().numpy(),
            "text_len": text_len.cpu().numpy(),
            "prompt_text": prompt_text.cpu().numpy(),
            "prompt_text_len": prompt_text_len.cpu().numpy(),
            "prompt_speech_token": prompt_speech_token.cpu().numpy(),
            "prompt_speech_token_len": prompt_speech_token_len.cpu().numpy(),
            "embedding": embedding.cpu().numpy(),
        }

        output = self.session.run(None, inputs)[0]  # (1, seq_len)
        for token in output[0]:
            yield int(token)
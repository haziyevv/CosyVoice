import torch
import torch.nn as nn

class Qwen2LMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, text_len, prompt_text, prompt_text_len,
                prompt_speech_token, prompt_speech_token_len, embedding):
        gen = self.model.inference(text, text_len, prompt_text, prompt_text_len,
                                   prompt_speech_token, prompt_speech_token_len,
                                   embedding)
        tokens = []
        for _ in range(200):  # or any reasonable fixed length
            try:
                tok = next(gen)
                tokens.append(tok)
            except StopIteration:
                break
        return torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)  # shape: (1, T)



 model = Qwen2LMWrapper(self.model.llm).cuda()
        model.eval()

        dummy_input = {
            "text": torch.randint(0, 100, (1, 10)).int().cuda(),
            "text_len": torch.tensor([10]).int().cuda(),
            "prompt_text": torch.randint(0, 100, (1, 5)).int().cuda(),
            "prompt_text_len": torch.tensor([5]).int().cuda(),
            "prompt_speech_token": torch.randint(0, 100, (1, 15)).int().cuda(),
            "prompt_speech_token_len": torch.tensor([15]).int().cuda(),
            "embedding": torch.randn(1, 896).cuda(),
        }
        torch.onnx.export(
            model,
            (dummy_input['text'], dummy_input['text_len'],
            dummy_input['prompt_text'], dummy_input['prompt_text_len'],
            dummy_input['prompt_speech_token'], dummy_input['prompt_speech_token_len'],
            dummy_input['embedding']),
            "qwen2lm.onnx",
            input_names=["text", "text_len", "prompt_text", "prompt_text_len",
                        "prompt_speech_token", "prompt_speech_token_len", "embedding"],
            output_names=["output"],
            opset_version=14,
            dynamic_axes={"text": {1: "text_len"}, "output": {1: "out_len"}},
            do_constant_folding=True
        )

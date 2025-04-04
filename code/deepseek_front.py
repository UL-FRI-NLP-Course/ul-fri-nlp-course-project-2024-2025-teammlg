from typing import Optional
import ollama


class DeepSeekChatBot:
    def __init__(self):
        self.model_label = "deepseek-r1:1.5b"  # The name of the model for Ollama to download (all models here: https://ollama.com/search)
        self._download_model_if_missing()

    def prompt(self, prompt: str) -> ollama.GenerateResponse:
        """Feeds the prompt to the model, returning its response"""
        return ollama.generate(model=self.model_label, prompt=prompt)

    def prompt_with_context(
        self, prompt: str, instructions: Optional[str], context: Optional[str]
    ) -> ollama.GenerateResponse:
        """
        For prompt engineering tasks. The method takes in the user prompt,
        and optionally instructions and context.

        The final prompt looks like this:
        ```
        Instructions: <instructions>

        Context: <context>

        <prompt>
        ```
        The method returns the response from the model.
        """
        final_prompt = ""
        if instructions:
            final_prompt += f"Instructions: {instructions}\n\n"
        if context:
            final_prompt += f"Context: {context}\n\n"
        final_prompt += prompt
        return ollama.generate(model=self.model_label, prompt=final_prompt)

    def _download_model_if_missing(self):
        """Checks if the model is already downloaded, and downloads it otherwise"""
        all_local_models = ollama.list()
        for model in all_local_models.models:
            if model.model == self.model_label:
                return  # We found the model - we exit
        print(f"Could not find local '{self.model_label}' instance, downloading...")
        response = ollama.pull(self.model_label)
        print(response.completed)


class LanguageModel:
    #
    # llm_config - path to file with model configuration
    #
    def __init__(self, config : str):
        pass
    
    #
    # Given strings with the user and system prompts, query the LLM
    # and return the response.
    #
    def generate_response(self, system_prompt : str, user_prompt : str) -> str:
        pass

    #
    # Given a dataset, fine-tune the LLM.
    #
    def train(self, data : list) -> None:
        pass

    #
    # Given the system and user prompts as strings, format them
    # into a list of messages formated as dictionaries.
    #
    # Example:
    #   
    #    Input:
    #       - system_prompt = "You are an AI agent in a maze"
    #       - user_prompt = "Given this game state, which of the following actions would you take..."
    #
    #    Output:
    #       - messages = [{'role': 'system', 'content': system_prompt},
    #                     {'role': 'user', 'content': user_prompt}]
    #
    def prompts_to_messages(self, system_prompt:str, 
                                  user_prompt:str,
                                  response:str=None) -> list[dict]:
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}]
        if response is not None:
            messages.append({'role': 'assistant', 'content': response})
        return messages
    
    def eval(self):
        self.model.eval()
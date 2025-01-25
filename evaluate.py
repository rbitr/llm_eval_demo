from abc import ABC, abstractmethod
from transformers import pipeline
from typing import List, Dict, Tuple
from statistics import mean
from tqdm import tqdm
import re
from collections import Counter

eps = 0.001

class LLM(ABC):
    """
    An abstract class to implement the functionality of different LLMs
    They all will have a "chat" function that takes a prompt and returns the
    LLM output.
    """
    def __init__(self, model : str="", 
                 max_tokens : int=512, 
                 temperature: float=0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @abstractmethod
    def chat(self, query : str) -> str:
        pass

class HF_LLM(LLM):
    """
    This is an example for HuggingFace language models hosted locally.
    Pass the name of the model, e.g. model="Qwen/Qwen2.5-0.5B-Instruct" when instantiating
    """
    def chat(self, query : str):
        messages = [{"role": "user", "content": query},]
        pipe = pipeline("text-generation",self.model)
        sample = False if self.temperature == 0 else True
        response = pipe(messages, max_new_tokens=self.max_tokens, do_sample=sample, temperature=max(self.temperature,eps))
        return response[0]["generated_text"][1]["content"]


class Feature(ABC):
    """
    Generic class to implement the calculation of a feature
    """
    @abstractmethod
    def __call__(self, model_outputs: List[str]) -> float | dict:
        pass

class NumberOfSentences(Feature):
    """ A simple demonstration feature that just counts the average number of sentences in responses
        based on the number of periods. 
    """
    def __call__(self, model_outputs: List[str]) -> float:
        return mean( [len([1 for c in output if c=="."]) for output in model_outputs] )

class BigramFreq(Feature):
    """
    Another example.
    Calculates the frequency (proportino of occurrence) of a specific bigram amoungs all bigrams
    present in a list of texts. 
    """
    def __init__(self, bigram: Tuple[str, str]):
        self.bigram = bigram
    
    def __call__(self, model_outputs: List[str]) -> float:
        total = 0
        count = 0
        for output in model_outputs:
            words = re.findall(r'\b\w+\b', output)
            words = [w.lower() for w in words]
            bigrams = list(zip(words, words[1:]))
            count += Counter(bigrams)[self.bigram]
            total += len(bigrams)

        return count/total

class Eval:
    """
    Evaluate a list of prompts, first by running inference to get a list of outputs, and then by
    calculating the features.
    An optional "filters" list can be used to only compute the features when the prompt matches some specific string.
    """
    def __init__(self, llm: LLM, prompts: List[str]=None, evals: List[Feature]=None, filters: List[str]=None):
        self.llm = llm
        self.prompts = [p for p in prompts] if prompts else []
        self.evals = [e for e in evals] if evals else []
        self.filters = [f for f in filters] if filters else [""]*len(evals)

    def inference(self,repeats=1):

        results = []
        for prompt in tqdm(self.prompts):
            result = {"prompt":prompt, "responses":[]}
            for repeat in range(repeats):
                response = self.llm.chat(prompt)
                result["responses"].append(response)
            results.append(result)

        self.results = results
        return results
    
    def compute_features(self):

        eval_results = []
        for result in self.results:
            prompt = result["prompt"]
            eval_result = {"prompt":prompt}
        
            features = []
            for e, f in zip(self.evals, self.filters):
                if not f or f in prompt.lower(): # if a filter is set, the filter phrase must be in the prompt
                    features.append({"feature":e.__class__.__name__, "value": e(result["responses"])})

            eval_result["features"] = features
            eval_results.append(eval_result)

        self.features = eval_results
        return eval_results

        

    
        

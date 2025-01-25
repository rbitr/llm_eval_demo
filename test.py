from evaluate import HF_LLM, Eval, NumberOfSentences, BigramFreq

with open("prompts.txt") as f:
    prompts = f.readlines()

features = [NumberOfSentences(), BigramFreq(("rolling", "hills"))]

llm = HF_LLM(model="Qwen/Qwen2.5-0.5B-Instruct",
	     max_tokens=128,
	     temperature=1)

# run the rilling hills bigram only on prompts about a story
llm_eval = Eval(llm, prompts=prompts, evals=features, filters=["", "story"])

print(llm_eval.inference(2))
print(llm_eval.compute_features())

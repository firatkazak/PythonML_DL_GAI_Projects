from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.generation_config.pad_token_id = model.generation_config.eos_token_id

input_prompt = "I enjoy walking with my cute dog"
model_inputs = tokenizer(input_prompt, return_tensors="pt")

print("Model Input'u: ", model_inputs)

greedy_output = model.generate(**model_inputs,
                               max_new_tokens=40,
                               )

greedy_output_result = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
print("greedy output result: ", greedy_output_result)

beam_output = model.generate(**model_inputs,
                             max_new_tokens=40,
                             num_beams=5,
                             early_stopping=True,
                             )

beam_output_result = tokenizer.decode(beam_output[0], skip_special_tokens=True)
print("beam output result: ", beam_output_result)

beam_output = model.generate(**model_inputs,
                             max_new_tokens=40,
                             num_beams=5,
                             early_stopping=True,
                             no_repeat_ngram_size=2,
                             )

beam_output_result_with_no_repeat = tokenizer.decode(beam_output[0], skip_special_tokens=True)
print("beam output result with no repeat: ", beam_output_result_with_no_repeat)

beam_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
)

for i, beam_output in enumerate(beam_output):
    print("with num return sequences: ", f"{i}, {tokenizer.decode(beam_output, skip_special_tokens=True)}")

set_seed(0)

sample_output = model.generate(**model_inputs,
                               max_new_tokens=40,
                               do_sample=True,
                               )

sample_output_result = tokenizer.decode(sample_output[0], skip_special_tokens=True, )
print("sample output result: ", sample_output_result)

sample_output = model.generate(**model_inputs,
                               max_new_tokens=40,
                               do_sample=True,
                               temperature=0.6,
                               )

sample_output_with_temperature = tokenizer.decode(sample_output[0], skip_special_tokens=True)
print("sample output with temperature: ", sample_output_with_temperature)

sample_output = model.generate(**model_inputs,
                               max_new_tokens=40,
                               do_sample=True,
                               top_k=50,
                               )

sample_output_with_top_k = tokenizer.decode(sample_output[0], skip_special_tokens=True)
print("sample output with top k: ", sample_output_with_top_k)

sample_output = model.generate(**model_inputs,
                               max_new_tokens=40,
                               do_sample=True,
                               top_k=0,
                               top_p=0.92
                               )

sample_output_with_top_p = tokenizer.decode(sample_output[0], skip_special_tokens=True)
print("sample output with top p: ", sample_output_with_top_p)

sample_output = model.generate(**model_inputs,
                               max_new_tokens=40,
                               do_sample=True,
                               top_k=50,
                               top_p=0.92,
                               num_return_sequences=3,
                               )

for i, sample_output in enumerate(sample_output):
    print("sample output with num_return_sequences: ", f"{i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")

import os, sys, string

#load the or segments defined table and get the seperated key word
def get_seprtword():
	s_app = ""
	s_req = ""
	s_gap = ""
	s_pro = ""
	s_val = ""
	s_chk = ""
	s_dep = ""

#input the raw or data text and seperate the sentences and embedding
def ld2emb_rawortext():
	encodes = getEncode()
    model   = getModel()
    model.layers[0].set_weights([getEmbedding()])

#execute the seperating generate model to seperate the raw or text by the key word 
def exec_seprtor():
	model_path = "./models/7B/ggml-model-q4_0"
	tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
	if model_path.endswith("4bit"):
    	model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map='auto')
	else:
    	model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
	streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
	prompt = instruction.format(text)
    generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)
    
#store the or text after seperated by the GPT in excel and .xml
def store_seprtdortxt():
	fs = file.open("")
	fs.close()
	
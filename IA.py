from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys
tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
#if torch.cuda.is_available():
#    model = model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

## utility functions ##
#flatten = lambda l: [item for sublist in l for item in sublist]

def flatten(lst):
    return [item for sublist in lst for item in sublist]


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line,clean_up_tokenization_spaces=False)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("> Your Npc: "+msg)
            print()

lvl = -1
while lvl not in [0, 1, 2, 3]:
    lvl = int(input("What level of creativity you want to allow your npc to be?\n1 - Creative \n2 - Normal \n3 - Precise \n0 - Quit\n"))

if lvl == 0:
    print("Exiting...")
    sys.exit()

# Define generation parameters based on selected level
if lvl == 1:
    top_k = 50
    top_p = 0.8
    temperature = 2
    print("NPC set to Creative level.")
elif lvl == 2:
    top_k = 10
    top_p = 0.9
    temperature = 1
    print("NPC set to Normal level.")
elif lvl == 3:
    top_k = 6
    top_p = 0.95
    temperature = 0.5
    print("NPC set to Precise level.")

def generate_next(bot_input_ids, top_k, top_p,temperature,max_length=1000, do_sample=True, pad_token=tokenizer.eos_token_id):
    # Create an attention mask
    attention_mask = bot_input_ids.ne(tokenizer.pad_token_id).long()

    # Pass the attention mask to the generate function
    full_msg = model.generate(
        bot_input_ids, 
        attention_mask=attention_mask,
        do_sample=do_sample, 
        top_k=top_k, 
        top_p=top_p, 
        temperature=temperature,
        max_length=max_length, 
        pad_token_id=pad_token
    )

    msg = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
    return msg



#============================pour donner des personnalités depuis le terminal===================#
personas = []
n_facts = int(input("How many personality facts do you want to give? Enter -1 for Default(3) "))
if n_facts == -1:
    n_facts=3
for i in range(n_facts):
    response = input(">> Fact %d: "%(i+1))+ tokenizer.eos_token
    personas.append(response)
personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))
#==============================================================================================#

#============================pour hardcoder des personnalités==================================#
# personas = ["My name is Rick", "I like chocolate"]
# for persona in personas:
#     persona += tokenizer.eos_token

# personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))
#==============================================================================================#


#==================================conversation sur X tours====================================#
dialog_hx = []
n_dialogs = int(input("How many prompts? Enter -1 for Default(8) "))
if n_dialogs == -1:
    n_dialogs=8
    
for step in range(n_dialogs):
    # encode the user input
    user_inp = tokenizer.encode(input(">> User: ") + tokenizer.eos_token)
    # append to the chat history
    dialog_hx.append(user_inp)
        
    # generated a response while limiting the total chat history to 1000 tokens, 
    bot_input_ids = to_var([personas + flatten(dialog_hx)]).long()
    msg = generate_next(bot_input_ids,top_k,top_p,temperature)
    dialog_hx.append(msg)
    print("> Your Npc: {}".format(tokenizer.decode(msg, skip_special_tokens=True, clean_up_tokenization_spaces=False)))
root_path='.'
from tqdm import tqdm
import json
import csv
import sys
sys.path.insert(0, "/hdd0/hyeon/dspy")
import os
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(root_path, 'cache')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from dsp.modules import *
import dsp
import ujson
from dsp.utils import deduplicate
colbert_server = 'http://163.152.163.176:8893/api/search'
lm = dsp.GPT3(model='gpt-4-turbo-preview', api_key=openai_key,model_type='chat', max_tokens = 4096)
rm = dsp.ColBERTv2(url=colbert_server)
dsp.settings.configure(lm=lm, rm=rm)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
test_file_path = "/hdd0/hyeon/mediqa/test_set_1.json"
outputfile= '/hdd0/hyeon/mediqa/gpt_1.json'

# Path to your original dataset
file_path = '/hdd0/hyeon/mediqa/cot/train_valid_vicuna_format.json'

with open(file_path, 'r') as file:
    data = json.load(file)



# Specify the path to your CSV file
file_path1 = '/hdd0/hyeon/mediqa/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-TrainingData.csv' #MS_train
file_path2 = "/hdd0/hyeon/mediqa/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv" #MSval
file_path3 = '/hdd0/hyeon/mediqa/Feb_5_2024_UW_Validation_Set_Updated/MEDIQA-CORR-2024-UW-ValidationSet-1-Full_Feb.csv' #UW_val

# Open the file in read mode
def load_csv(file_path):
    with open(file_path, mode='r') as file:
        mediqa = []
        # Create a CSV reader
        csv_reader = csv.reader(file)
        
        # Skip the header if there is one
        next(csv_reader)
        
        # Iterate over the CSV rows
        for i, row in enumerate(csv_reader):
            mediqa.append(row)
    return mediqa

mediqa= load_csv(file_path1) + load_csv(file_path2) + load_csv(file_path3)

id_mediqa = {}
for i in mediqa:
    _id = i[1]
    text = i[2]
    errorflag = i[4]
    error_sen = i[6]
    corr_sen = i[7]
    id_mediqa[_id] = {'text': text, 'error_flag':errorflag, 'error_sentence': error_sen, 'corrected_sentence' : corr_sen}

for i in data:
    id_ori = '-'.join(i['id'].split('-')[:-1])
    i['output']= id_mediqa[id_ori]

question_answer_pairs = []
for inst in data:
    _id = inst['id']
    text = inst['output']['text']
    full = inst['output']['text'] + "Rationale:\n"+ inst['conversations'][1]['value'] + "\nError Sentence: " + inst['output']['error_sentence'] + "\nCorrected Sentence: " + inst['output']['corrected_sentence']
    rationale = inst['conversations'][1]['value']
    error_sen = inst['output']['error_sentence']
    corr_sen = inst['output']['corrected_sentence']
    question_answer_pairs.append((_id,full, text, rationale, error_sen, corr_sen))


train_pairs = [dsp.Example(id=_id, question=full, text=text, rationale=rationale, error_sen=error_sen,corrected_sen=corr_sen) for _id,full, text, rationale, error_sen, corr_sen in question_answer_pairs]


# set new vectorizer for all calls
dsp.settings.configure(vectorizer=dsp.SentenceTransformersVectorizer())
knn_func = dsp.knn(train_pairs)





# Open the file in read mode
with open(test_file_path) as f:
    test = json.load(f)

dev = [dsp.Example(_id= inst['text_id'], question=inst['text']) for inst in test]

Text = dsp.Type(prefix="Text:", desc="${The text to be corrected}")
Error_sen = dsp.Type(prefix="Error Sentence:", desc="${The sentence that contains the error. Enter 'NA' if there's no error in the text.}")
Corrected_sen = dsp.Type(prefix="Corrected Sentence:", desc="${A corrected version of the sentence that contains the error, altering only the specific phrase that is incorrect. Enter 'NA' if there's no error in the text.}")
Rationale = dsp.Type(
    prefix="Rationale: Let's think step by step.",
    desc="${a step-by-step deduction that identifies the correct response, which will be provided below}"
)
qa_template = dsp.Template(instructions="""You are an expert tasked with providing a logical explanation as to whether there is an error in the given clinical note. Your job is to analyze the clinical note step-by-step and and provide an explanation leading to the conclusion regarding the presence or absence of an error. Typical errors include incorrect medication or dosage, misdiagnoses, overlooking essential patient history or details, and procedural mistakes in patient care or treatment plans. Note that within the clinical note, there is either one error or none at all. It is important to focus on correcting only the erroneous phrase, if any, without making extensive corrections.""", text=Text(), rationale=Rationale(),error_sen=Error_sen(),corrected_sen=Corrected_sen())

@dsp.transformation
def QA_predict(example: dsp.Example):
    example, completions, output = dsp.generate(qa_template, temperature=1.0)(example, stage='qa')
    
    return example, output


def deduplicate_id(entries):
    seen = set()  # Keep track of seen "xxx" parts
    output = []
    for item in entries:
        # Extract the "xxx" part from the "id"
        id_parts = item['id'].split('-')
        unique_id = '-'.join(id_parts[:-1])  # This combines 'ms-train' and 'xxx'

        if unique_id not in seen:
            seen.add(unique_id)
            output.append(item)
    
    return output


def Predict(example: str) -> str:
    knn_res_train_vec = knn_func(example, 25)
    knn_res_train_vec = deduplicate_id(knn_res_train_vec)[:5]
    knn_shot=[dsp.Example(text=qa['text'], rationale=qa['rationale'], error_sen=qa['error_sen'], corrected_sen=qa['corrected_sen']) for qa in knn_res_train_vec]
    demos = dsp.sample(knn_shot, k=5)
    x = dsp.Example(text=example.question, demos=demos)
    x, output = QA_predict(x)

    return x, output


for i in tqdm(range(len(test))):
    x, output = Predict(dev[i])
    test[i]['prediction'] = output[0]
# Open a text file for writing

with open(outputfile, 'w') as file:
    json.dump(test, file)
    # Loop through your data with a progress bar



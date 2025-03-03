import sacrebleu
from rouge_score import rouge_scorer
import nltk
from nltk.translate import meteor_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import pandas as pd

# Download necessary NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def BLEU(candidate,reference):
    return sacrebleu.sentence_bleu(candidate, [reference]).score

def ROUGE(candidate,reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)['rougeL'].fmeasure

    return rouge_scores

def METEOR(candidate, reference):
    meteor = meteor_score.single_meteor_score(reference.split(), candidate.split())

    return meteor

def EM(candidate, reference):
    if reference == candidate:
        return 1
    else:
        return 0
    
    
def process_input(question, answer, prediction):
    premise = 'question: '+question+' '+'answer: '+answer
    hypothesis = 'question: '+question+' '+'answer: '+prediction

    return premise,hypothesis

def get_semantic_similarity(question, answer, prediction,model,tokenizer,device='cuda'):
    
    premise, hypothesis = process_input(question, answer, prediction)


    # Tokenize input 1
    
    input_1 = tokenizer( hypothesis,premise ,truncation=True, return_tensors="pt")
    input_1 = {key: value.to(device) for key, value in input_1.items()}

    # Model prediction for input 1
    output_1 = model(**input_1)
    prediction_1 = torch.softmax(output_1["logits"][0], -1).tolist()
    if model.name_or_path == 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli':
        label_names = ["entailment","neutral","contradiction"]
    else:
        label_names = ["entailment","contradiction"]
    
    prediction_1 = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction_1, label_names)}

    _ = {key: value.cpu() for key, value in input_1.items()}
    del input_1
    torch.cuda.empty_cache()
    
    return (prediction_1['entailment']/100)

def get_lexical_similarity(answer, prediction):
    answer_list = answer.split(" ")
    count = 0
    for word in prediction.split(" "):
        if word in answer_list:
            count += 1
    return count/len(answer_list)

def initialize_model(device='cuda'):
    models = ['MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli']
    tokenizer = AutoTokenizer.from_pretrained(models[0])
    model = AutoModelForSequenceClassification.from_pretrained(models[0]).to(device)
    return model,tokenizer


def evaluator(X,log_regr_model,device='cuda'):
    models = ['MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli']
    tokenizer = AutoTokenizer.from_pretrained(models[0])
    model = AutoModelForSequenceClassification.from_pretrained(models[0]).to(device)
    lex_eval = X.apply(lambda x: get_lexical_similarity(x[X.columns[1]],x[X.columns[2]]), axis=1)
    sem_eval = X.apply(lambda x: get_semantic_similarity(x[X.columns[0]],x[X.columns[1]],x[X.columns[2]],model,tokenizer), axis=1)
    x = pd.DataFrame({'lex_eval':lex_eval,'sem_eval': sem_eval})
    pred = log_regr_model.predict(x)
    return pred

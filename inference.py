# inference.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def load_finetuned_model(model_dir):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    # Setup the text generation pipeline
    text_generation_pipeline = pipeline(
        'text-generation', 
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available
    )

    return text_generation_pipeline

def ask_question(pipeline, question, context='학생들을 위한 진로상담가', max_tokens=512, temperature=0.7, top_p=0.9):
    prompt = (
        f"### 질문: {question}\n\n### 역할: {context}\n\n### 답변:" 
        if context else f"### 질문: {question}\n\n### 답변:"
    )
    
    # Generate the answer using the pipeline
    response = pipeline(
        prompt, 
        do_sample=True, 
        max_new_tokens=max_tokens,  # Adjust answer length here
        temperature=temperature,
        top_p=top_p,
        eos_token_id=pipeline.tokenizer.eos_token_id,
    )
    
    # Extract and print the generated answer
    generated_text = response[0]['generated_text']
    answer = generated_text.split("### 답변:")[-1].strip()  # Extract the answer part
    return answer

if __name__ == "__main__":
    # Load the model
    pipeline = load_finetuned_model("EleutherAI/polyglot-ko-3.8b")

    # Example usage
    question = "진로 선택에 대해 고민하고 있습니다. 도움을 줄 수 있을까요? 평소 좋아하는 취미생활은 그림 그리기입니다. 평소 좋아하는 과목은 미술입니다. 제 전시를 사람들이 종아해줄 때 행복해요."
    answer = ask_question(pipeline, question)
    print(f"Q: {question}\nA: {answer}\n")
    
    
    
{
    consult : "그림에 관심이 있다니 미술 쪽으로 진로를 잡으면 좋겠어요.~~~~~~",
    jobs_list : {1: 미술가, 2: ~~~~}.
}

# llama-cpp-python 이 설치되어 있지 않다면 아래 주석을 해제하여 설치합니다.
# !pip install llama-cpp-python 

# q4_0 모델을 Files 탭에서 직접 다운로드 하거나 아래 주석을 해제하여 다운로드 합니다.
# !pip install huggingface_hub #
# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id='StarFox7/Llama-2-ko-7B-chat-ggml', filename='Llama-2-ko-7B-chat-ggml-q4_0.bin', local_dir='./')

from llama_cpp import Llama

llm = Llama(model_path = 'models/Llama-2-ko-7B-chat-gguf-q4_0.bin',
            n_ctx=1024,
            n_gpu_layers=256
            
            #gpu 가속을 원하는 경우 주석을 해제하고 Metal(Apple M1) 은 1, Cuda(Nvidia) 는 Video RAM Size 를 고려하여 적정한 수치를 입력합니다.
      )
output = llm("Q: 인생이란 뭘까요?. A: ", max_tokens=1024, stop=["Q:", "\n"], echo=True)
print( output['choices'][0]['text'].replace('▁',' ') )
#출력 결과
'''
Q: 인생이란 뭘까요?. A: 30,000개의 미생물이 사는 장 속의 세균 같은 것. 
'''

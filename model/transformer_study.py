import torch # Pytorch
import sys # python 시스템 모듈 : 프로그램 종료, 경로 등
import signal # 신호 처리 ex) SIGINT - 프로그램 중간에 종료 시
import torch.nn.functional as F # 신경망 함수 - 활성화 함수, 손실 함수 제공
from torch import tensor, Tensor # 텐서 생성, 클래스
from typing import List, Dict, Tuple # 함수, 변수에 타입 명시
from tqdm import tqdm # 프로세스 진행 상태 표시

### CUDA SETUP ###

if torch.cuda.is_available():  # CUDA(NVIDIA GPU 가속)가 시스템에서 사용 가능한지 확인
    device = torch.device('cuda')  # GPU(CUDA)를 사용할 수 있으면 장치를 CUDA로 설정
    torch.set_default_device(device)  # CUDA를 모든 텐서와 연산의 기본 장치로 설정
    print(f"Using {torch.cuda.get_device_name()}")  # 사용 중인 GPU의 이름을 출력
else:  # CUDA를 사용할 수 없는 경우
    device = torch.device('cpu')  # 장치를 CPU로 설정
    torch.set_default_device(device)  # CPU를 모든 텐서와 연산의 기본 장치로 설정
    print("Using CPU")  # CPU를 연산 장치로 사용하는 것을 알림


### NEURAL NETWORK OBJECTS ###

class LinearLayer:  # 선형 계층(Linear Layer) 클래스 정의
    def __init__(self, input_size: int, output_size: int):  
        # 생성자: 입력 크기와 출력 크기를 기반으로 가중치와 편향 초기화
        self.weights = torch.randn(input_size, output_size, dtype=torch.float64, requires_grad=False) * 0.01  
        # 가중치를 작은 난수로 초기화, 역전파 시 텐서 기울기 자동계산 X(requires_grad=False)
        self.bias = torch.zeros(output_size, dtype=torch.float64, requires_grad=False)  
        # bias을 0으로 초기화
        self.grad_weights = torch.zeros_like(self.weights)  
        # 가중치의 기울기를 0으로 초기화
        self.grad_bias = torch.zeros_like(self.bias)  
        # 편향의 gradient를 0으로 초기화

    def forward(self, x: Tensor) -> Tensor:  
        # 순전파(Forward pass) 메서드: 입력 텐서를 선형 변환
        self.input = x  # 입력 데이터를 저장 (역전파에 사용됨)
        return x @ self.weights + self.bias  # 선형 연산 수행 (xW + b) - @ : 행렬 곱셈

    def backward(self, grad_output: Tensor) -> Tensor:  
        # 역전파(Backward pass) 메서드: 출력의 기울기를 입력과 가중치의 기울기로 변환
        self.grad_weights += self.input.transpose(0, 1) @ grad_output  
        # weight 가중치 계산 : 입력의 전치와 출력 기울기를 곱하여 가중치의 기울기를 계산하고 누적 - transpose(0,1) : 0,1번째 순서를 바꿈(2차원에서는 .T와 동일)
        self.grad_bias += grad_output.sum(dim=0)  
        # bias 가중치 계산 : 출력 기울기를 더하여 편향의 기울기를 계산하고 누적
        grad_input = grad_output @ self.weights.transpose(0, 1)  
        # input(x) 가중치 계산 : 출력 기울기와 가중치의 전치를 곱하여 입력 기울기를 계산
        return grad_input  # 입력 기울기를 반환

class ReLUActivation:  # ReLU (Rectified Linear Unit) 활성화 함수 클래스 정의
    def forward(self, x: Tensor) -> Tensor:  
        # 순전파 메서드: ReLU 활성화 함수 적용
        self.input = x  # 입력 데이터를 저장 (역전파에서 사용)
        return torch.clamp(x, min=0)  # ReLU 연산: 입력 값이 0(min)보다 작으면 min값으로 설정 - clamp : 함수 max보다 큰 경우 or min보다 작은 경우 가능

    def backward(self, grad_output: Tensor) -> Tensor:  
        # 역전파 메서드: ReLU의 기울기 계산
        grad_input = grad_output.clone()  # 출력 기울기를 복사하여 입력 기울기 생성
        grad_input[self.input <= 0] = 0  # 입력 값이 0 이하인 경우 기울기를 0으로 설정 - relu 기울기 적용
        return grad_input  # 계산된 입력 기울기 반환

class GeLUActivation:  # GeLU (Gaussian Error Linear Unit) 활성화 함수 클래스 정의
    def forward(self, x: Tensor) -> Tensor:  
        # 순전파 메서드: GeLU 활성화 함수 적용
        self.input = x  # 입력 데이터를 저장 (역전파에서 사용)
        c = torch.tensor(0.7978845608, dtype=x.dtype, device=x.device)  # device : cpu or gpu
        # 상수 c = sqrt(2/pi)
        a = torch.tensor(0.044715, dtype=x.dtype, device=x.device)  
        # 상수 a = 2/(pi*sqrt(2))
        self.c = c  # 역전파에서 사용할 c 저장
        self.a = a  # 역전파에서 사용할 a 저장
        self.s = c * (x + a * x ** 3)  # 중간 계산값 s = c * (x + a * x^3)
        self.tanh_s = torch.tanh(self.s)  # tanh(s)를 계산
        y = 0.5 * x * (1.0 + self.tanh_s)  # GeLU 연산: y = 0.5 * x * (1 + tanh(s))
        return y  # GeLU 활성화 결과 반환

    def backward(self, grad_output: Tensor) -> Tensor:  
        # 역전파 메서드: GeLU의 기울기 계산 >> x에 대해 1번, s에 대해 한 번
        x = self.input  # 입력 값을 가져옴
        c = self.c  # 상수 c 가져옴
        a = self.a  # 상수 a 가져옴
        s = self.s  # 중간 값 s 가져옴
        # --------------- y에 대한 미분 ---------------------      
        tanh_s = self.tanh_s  # 중간 값 tanh(s) 가져옴
        sech2_s = 1.0 - tanh_s ** 2  # tanh의 도함수 계산: sech^2(s) = 1 - tanh(s)^2
        ds_dx = c * (1.0 + 3.0 * a * x ** 2)  
        # s의 x에 대한 도함수: ds/dx = c * (1 + 3 * a * x^2) - 속미분 값       
        dy_dx = 0.5 * (1.0 + tanh_s + x * sech2_s * ds_dx)  
        # y의 x에 대한 도함수: dy/dx = 0.5 * (1 + tanh(s) + x * sech^2(s) * ds/dx)
        grad_input = grad_output * dy_dx  # 입력 기울기 계산: grad_input = grad_output * dy/dx - x에 대한 미분값으로 x를 업데이트해서 흘려주기
        return grad_input  # 계산된 입력 기울기 반환

class Embedding:  # Embedding 클래스: 단어/토큰을 고차원 벡터로 매핑하는 역할
    def __init__(self, input_dim: int, output_dim: int) -> None:
        # 생성자: 임베딩 테이블 초기화
        self.input_dim = input_dim  # 입력 차원 (단어 집합 크기 등)
        self.output_dim = output_dim  # 출력 차원 (임베딩 벡터 크기)
        self.weights: Tensor = torch.randn(input_dim, output_dim, dtype=torch.float64, requires_grad=False) * 0.01  
        # 임베딩 테이블을 난수로 초기화 (학습되지 않음)
        self.grad_weights: Tensor = torch.zeros_like(self.weights)  
        # 임베딩 테이블의 기울기를 0으로 초기화

    def forward(self, input_indices: Tensor) -> Tensor:
        # 순전파 메서드: 주어진 인덱스에 따라 임베딩 벡터 반환
        self.input_indices = input_indices  # 입력된 인덱스를 저장
        self.output = self.weights[input_indices]  # 인덱스를 기반으로 임베딩 테이블에서 벡터 추출
        return self.output  # 추출된 임베딩 벡터 반환

    def backward(self, grad_output: Tensor) -> None:
        # 역전파 메서드: 임베딩 테이블의 기울기 계산
        grad_flat = grad_output.view(-1, self.output_dim) # 임베딩 차원  
        # 출력 기울기를 (3D -> 2D)로 평탄화
        input_flat = self.input_indices.view(-1) # 단어 차원
        # 입력 인덱스를 평탄화
        self.grad_weights.index_add_(0, input_flat, grad_flat)  # 행(0) 기준으로 input_flat의 각 인덱스에 대해 grad_flat의 값을 grad_weights에 더함

        # 각 인덱스에 대해 기울기를 누적(index_add_ 사용)

class PositionalEncoding:  # PositionalEncoding 클래스: 위치 정보를 임베딩에 추가
    def __init__(self, max_seq_len: int, embed_size: int):
        # 생성자: 위치 인코딩 테이블 생성
        self.embed_size = embed_size  # 임베딩 크기
        self.pos_encoding = torch.zeros(max_seq_len, embed_size, dtype=torch.float64, requires_grad=False)  
        # 위치 인코딩 테이블 초기화 (0으로)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)   
        # 위치 인덱스 (0, 1, 2, ..., max_seq_len-1)를 열 벡터로 생성 - pos값 생성
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-torch.log(torch.tensor(10000.0)) / embed_size))  
        # positional encoding 공식에서 sin,cos 안의 식
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)  
        # 짝수 차원: sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)  
        # 홀수 차원: cos(position * div_term)

    def forward(self, x: Tensor) -> Tensor:
        # 순전파 메서드: 입력에 위치 인코딩 추가
        seq_length, embed_size = x.shape  # 입력 시퀀스 길이와 임베딩 크기
        pos_encoding = self.pos_encoding[:seq_length, :]  
        # 현재 시퀀스 길이에 맞는 위치 인코딩 추출
        return x + pos_encoding.to(x.device)  
        # 입력 텐서에 위치 인코딩을 더한 결과 반환 (같은 디바이스로 이동)

class MultiHeadAttention:
    def __init__(self, embed_size: int, heads: int):
        # 초기화 메서드: 임베딩 차원과 헤드 개수를 설정
        self.embed_size = embed_size  # 전체 임베딩 차원
        self.heads = heads  # 멀티헤드 개수
        self.head_dim = embed_size // heads  # 각 헤드의 임베딩 차원

        # 임베딩 차원이 헤드 개수로 나누어 떨어지지 않으면 오류 발생
        if embed_size % heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads")

        # 각 헤드에 대해(해드 단위로 쪼개기) Query, Key, Value, Output의 가중치 초기화 - len(리스트) = 헤드 개수
        self.W_Q = [torch.randn(embed_size, self.head_dim, dtype=torch.float64, requires_grad=False) * 0.01 for _ in range(heads)]
        self.W_K = [torch.randn(embed_size, self.head_dim, dtype=torch.float64, requires_grad=False) * 0.01 for _ in range(heads)]
        self.W_V = [torch.randn(embed_size, self.head_dim, dtype=torch.float64, requires_grad=False) * 0.01 for _ in range(heads)]
        self.W_O = [torch.randn(self.head_dim, embed_size, dtype=torch.float64, requires_grad=False) * 0.01 for _ in range(heads)]

        # 각 헤드에 대해(해드 단위로 쪼개기) 가중치의 기울기(gradient) 초기화
        self.grad_W_Q = [torch.zeros_like(w) for w in self.W_Q]
        self.grad_W_K = [torch.zeros_like(w) for w in self.W_K]
        self.grad_W_V = [torch.zeros_like(w) for w in self.W_V]
        self.grad_W_O = [torch.zeros_like(w) for w in self.W_O]

    def forward(self, x: Tensor) -> Tensor:
        # 순전파 메서드: 입력 텐서 x 처리
        self.x = x  # 역전파를 위해 입력 저장
        seq_length, _ = x.size()  # 입력 시퀀스 길이와 차원 가져오기
        self.Q_heads, self.K_heads, self.V_heads = [], [], []  # 각 헤드의 Query, Key, Value 저장소 초기화

        # 각 헤드별로 Query, Key, Value 계산
        for i in range(self.heads):
            Q = x @ self.W_Q[i]  # Query 계산
            K = x @ self.W_K[i]  # Key 계산
            V = x @ self.W_V[i]  # Value 계산
            self.Q_heads.append(Q)
            self.K_heads.append(K)
            self.V_heads.append(V)

        # 자기회귀적(attention masking) 마스크 생성 - 각 원소가 -inf인 윗 대각행렬
        mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)
        self.head_outputs = []  # 각 헤드의 출력 저장소
        self.attention_weights = []  # 각 헤드의 어텐션 가중치 저장소
        self.scores = []  # 각 헤드의 스코어 저장소

        for i in range(self.heads):
            # 어텐션 스코어 계산
            scores = self.Q_heads[i] @ self.K_heads[i].transpose(0, 1) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
            scores += mask  # 마스크 적용 -> soft max 들어가기 직전 값
            attn_weights = torch.softmax(scores, dim=-1)  # 어텐션 가중치 계산
            self.attention_weights.append(attn_weights)
            self.scores.append(scores)
            attn_output = attn_weights @ self.V_heads[i]  # 어텐션 결과 계산
            head_output = attn_output @ self.W_O[i]  # 헤드별 최종 출력 계산 >> 임베딩 차원으로 넓혀주는 역할
            self.head_outputs.append(head_output)

        self.out = sum(self.head_outputs)  # 모든 헤드의 출력을 합산
        return self.out  # 최종 출력 반환

    def backward(self, grad_output: Tensor) -> Tensor:
        # 역전파 메서드: 출력 기울기를 기반으로 입력 기울기 계산
        grad_x = torch.zeros_like(self.x)  # 입력 기울기 초기화
        for i in range(self.heads):
            # 모든 헤드에 동일한 출력 기울기 사용
            grad_head_output = grad_output  # 각 헤드 출력에 대한 기울기

            attn_output = self.attention_weights[i] @ self.V_heads[i]  # 어텐션 출력 계산 >> 헤드 값의 attention score 가져오기
            
            # W_O[i]에 대한 기울기 계산
            self.grad_W_O[i] += attn_output.transpose(0, 1) @ grad_head_output  # W_O 기울기 계산 - attn_output.transpose(0, 1) : 계수

            # 어텐션 출력에 대한 기울기 계산 >> 가중치 누적시키기
            grad_attn_output = grad_head_output @ self.W_O[i].transpose(0, 1)

            # 어텐션 가중치와 Value에 대한 기울기 계산 - attention score 역산
            grad_attn_weights = grad_attn_output @ self.V_heads[i].transpose(0, 1)
            grad_V = self.attention_weights[i].transpose(0, 1) @ grad_attn_output

            # 소프트맥스 역전파 계산
            attn_weights = self.attention_weights[i] # - attention score
            grad_scores = attn_weights * (grad_attn_weights - (attn_weights * grad_attn_weights).sum(dim=-1, keepdim=True))

            # Query와 Key에 대한 기울기 계산
            scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # 스코어 스케일링
            grad_Q = grad_scores @ self.K_heads[i] * scale  # Query 기울기 계산
            grad_K = grad_scores.transpose(0, 1) @ self.Q_heads[i] * scale  # Key 기울기 계산

            # 가중치에 대한 기울기 계산
            self.grad_W_Q[i] += self.x.transpose(0, 1) @ grad_Q  # W_Q 기울기 계산
            self.grad_W_K[i] += self.x.transpose(0, 1) @ grad_K  # W_K 기울기 계산
            self.grad_W_V[i] += self.x.transpose(0, 1) @ grad_V  # W_V 기울기 계산

            # 입력에 대한 기울기 합산
            grad_x += grad_Q @ self.W_Q[i].transpose(0, 1)
            grad_x += grad_K @ self.W_K[i].transpose(0, 1)
            grad_x += grad_V @ self.W_V[i].transpose(0, 1)

        return grad_x  # 입력 기울기 반환
    
class LayerNorm:
    def __init__(self, embed_size: int, eps: float = 1e-5):
        # LayerNorm 초기화
        self.gamma = torch.ones(embed_size)  # 스케일 매개변수 (초기값 1)
        self.beta = torch.zeros(embed_size)  # 이동 매개변수 (초기값 0)
        self.grad_gamma = torch.zeros_like(self.gamma)  # gamma의 기울기 초기화
        self.grad_beta = torch.zeros_like(self.beta)  # beta의 기울기 초기화
        self.eps = eps  # 안정성을 위한 작은 값 (epsilon)

    def forward(self, x: Tensor) -> Tensor:
        # 순전파: 입력 x를 정규화하고 스케일/이동 적용
        self.x = x  # 입력 저장
        self.mean = x.mean(dim=-1, keepdim=True)  # 입력의 평균 계산
        self.var = x.var(dim=-1, unbiased=False, keepdim=True)  # 입력의 분산 계산
        self.std = torch.sqrt(self.var + self.eps)  # 표준편차 계산 (epsilon 추가)
        self.x_hat = (x - self.mean) / self.std  # 정규화된 입력 계산
        out = self.gamma * self.x_hat + self.beta  # 스케일 및 이동 적용
        return out  # 최종 출력 반환

    def backward(self, grad_output: Tensor) -> Tensor:
        # 역전파: 출력 기울기를 기반으로 입력 기울기 계산
        N, D = self.x.shape  # 입력 크기 가져오기
        x_mu = self.x - self.mean  # 입력과 평균의 차이
        std_inv = 1.0 / self.std  # 표준편차의 역수

        # gamma와 beta의 기울기 계산
        self.grad_gamma += torch.sum(grad_output * self.x_hat, dim=0)  # gamma의 기울기
        self.grad_beta += torch.sum(grad_output, dim=0)  # beta의 기울기

        # 정규화된 입력(x_hat)에 대한 기울기 계산
        dx_hat = grad_output * self.gamma

        # 분산(variance)에 대한 기울기 계산
        dvar = torch.sum(dx_hat * x_mu * -0.5 * std_inv.pow(3), dim=-1, keepdim=True) #keepdim : 차원유지, .pow : 3제곱

        # 평균(mean)에 대한 기울기 계산
        dmu = torch.sum(-dx_hat * std_inv, dim=-1, keepdim=True) + dvar * torch.mean(-2.0 * x_mu, dim=-1, keepdim=True)

        # 입력(x)에 대한 기울기 계산
        dx = (dx_hat * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)
        return dx  # 입력 기울기 반환

    
class AttentionBlock:
    def __init__(self, embed_size: int, heads: int):
        # AttentionBlock 초기화: MultiHeadAttention과 LayerNorm 초기화
        self.attention: MultiHeadAttention = MultiHeadAttention(embed_size, heads)  # 멀티헤드 어텐션
        self.layer_norm: LayerNorm = LayerNorm(embed_size)  # Layer Normalization

    def forward(self, x: Tensor) -> Tensor:
        # 순전파: 입력 x 처리
        self.x = x  # 역전파를 위해 입력 저장
        attention_out = self.attention.forward(x)  # 멀티헤드 어텐션 수행
        out = self.layer_norm.forward(attention_out + x)  # 잔여 연결(residual connection) 및 LayerNorm 수행
        return out  # 최종 출력 반환

    def backward(self, grad_output: Tensor) -> Tensor:
        # 역전파: 출력 기울기를 기반으로 입력 기울기 계산

        # LayerNorm의 역전파 수행
        grad_norm = self.layer_norm.backward(grad_output)

        # 잔여 연결의 기울기 처리
        grad_attention = grad_norm.clone()  # 어텐션 출력 기울기 복사
        grad_x = grad_norm.clone()  # 입력 x의 기울기 복사

        # 멀티헤드 어텐션의 역전파 수행
        grad_attention = self.attention.backward(grad_attention)

        # 잔여 연결에서 기울기 추가
        grad_x += grad_attention

        return grad_x  # 입력 기울기 반환


class FeedForward:
    def __init__(self, embed_size: int, forward_expansion: int):
        # FeedForward 네트워크 초기화
        # Linear -> Activation -> Linear 구조로 구성
        self.fc1 = LinearLayer(embed_size, embed_size * forward_expansion)  # 첫 번째 선형 레이어
        self.activation = GeLUActivation()  # 활성화 함수 (GeLU)
        self.fc2 = LinearLayer(embed_size * forward_expansion, embed_size)  # 두 번째 선형 레이어

    def forward(self, x: Tensor) -> Tensor:
        # 순전파: 입력 x 처리
        self.x = x  # 역전파를 위해 입력 저장
        out = self.fc1.forward(x)  # 첫 번째 선형 변환
        out = self.activation.forward(out)  # 활성화 함수 적용
        out = self.fc2.forward(out)  # 두 번째 선형 변환
        return out  # 출력 반환

    def backward(self, grad_output: Tensor) -> Tensor:
        # 역전파: 출력 기울기를 기반으로 입력 기울기 계산
        grad = self.fc2.backward(grad_output)  # 두 번째 레이어 역전파
        grad = self.activation.backward(grad)  # 활성화 함수 역전파
        grad = self.fc1.backward(grad)  # 첫 번째 레이어 역전파
        return grad  # 입력 기울기 반환

class OutputProjection:
    def __init__(self, embed_size: int, vocab_size: int):
        # Output Projection 초기화
        # 입력 임베딩을 어휘 크기(vocab size)로 변환
        self.W = torch.randn(embed_size, vocab_size, dtype=torch.float64, requires_grad=False) * 0.01  # 가중치 초기화
        self.grad_W = torch.zeros_like(self.W)  # 가중치 기울기 초기화

    def forward(self, x: Tensor) -> Tensor:
        # 순전파: 입력 x를 처리하여 어휘 확률 로짓 계산
        self.input = x  # 입력 저장
        self.logits = x @ self.W  # 행렬 곱 수행
        return self.logits  # 로짓 반환

    def backward(self, grad_output: Tensor) -> Tensor:
        # 역전파: 출력 기울기를 기반으로 가중치와 입력 기울기 계산
        self.grad_W += self.input.transpose(0, 1) @ grad_output  # 가중치 기울기 계산
        grad_input = grad_output @ self.W.transpose(0, 1)  # 입력 기울기 계산
        return grad_input  # 입력 기울기 반환

class AdamOptimizer:
    def __init__(self, params: Dict[str, Tensor], grads: Dict[str, Tensor], lr: float = 1e-3, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        """
        Adam Optimizer 초기화

        Args:
            params (Dict[str, Tensor]): 최적화할 매개변수 사전
            grads (Dict[str, Tensor]): 매개변수에 대응하는 기울기 사전
            lr (float): 학습률
            betas (Tuple[float, float]): 모멘텀 계수 (1차, 2차)
            eps (float): 수치적 안정성을 위한 작은 값
        """
        self.lr = lr  # 학습률
        self.beta1, self.beta2 = betas  # 1차, 2차 모멘텀 계수
        self.eps = eps  # 수치적 안정성을 위한 작은 값
        self.params = params  # 최적화할 매개변수
        self.grads = grads  # 매개변수에 대응하는 기울기
        self.m = {key: torch.zeros_like(param) for key, param in self.params.items()}  # 1차 모멘텀 초기화
        self.v = {key: torch.zeros_like(param) for key, param in self.params.items()}  # 2차 모멘텀 초기화
        self.t = 0  # 시간 단계 (step count) 초기화

    def step(self):
        """
        매개변수 업데이트 수행
        """
        self.t += 1  # 시간 단계 증가
        for key in self.params.keys():
            grad = self.grads.get(key)  # 매개변수에 대한 기울기 가져오기
            if grad is None:
                continue  # 기울기가 없으면 건너뜀

            # 1차 모멘텀 추정치 업데이트
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad

            # 2차 모멘텀 추정치 업데이트
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # 1차 모멘텀 추정치의 편향 보정
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)

            # 2차 모멘텀 추정치의 편향 보정
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 매개변수 업데이트
            self.params[key] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        모든 기울기를 0으로 초기화
        """
        for grad in self.grads.values():
            grad.zero_()  # 기울기를 0으로 초기화

class TransformerEncoderBlock:
    def __init__(self, embed_size: int, heads: int, ff_expand_dim: int):
        # Transformer Encoder Block 초기화
        self.attention = AttentionBlock(embed_size, heads)  # 멀티헤드 어텐션 블록
        self.layer_norm_1 = LayerNorm(embed_size)  # 첫 번째 LayerNorm
        self.feed_forward = FeedForward(embed_size, ff_expand_dim)  # 피드포워드 네트워크
        self.layer_norm_2 = LayerNorm(embed_size)  # 두 번째 LayerNorm

    def forward(self, x: Tensor) -> Tensor:
        # 순전파: 입력 x 처리
        self.x = x  # 역전파를 위해 입력 저장

        # 멀티헤드 어텐션 처리
        attention_out = self.attention.forward(x)

        # Residual connection (입력 x와 어텐션 출력의 합)
        x_residual = x + attention_out

        # 첫 번째 LayerNorm 적용
        x_norm = self.layer_norm_1.forward(x_residual)

        # Feed Forward Network 처리
        feed_forward_out = self.feed_forward.forward(x_norm)

        # Residual connection (첫 번째 LayerNorm 출력과 Feed Forward 출력의 합)
        x_ff_residual = x_norm + feed_forward_out

        # 두 번째 LayerNorm 적용
        out = self.layer_norm_2.forward(x_ff_residual)

        return out  # 최종 출력 반환

    def backward(self, grad_output: Tensor) -> Tensor:
        # 역전파: 출력 기울기를 기반으로 입력 기울기 계산

        # 두 번째 LayerNorm의 역전파
        grad_norm_2 = self.layer_norm_2.backward(grad_output)

        # Feed Forward 출력과 Residual 연결의 기울기 처리
        grad_feed_forward = grad_norm_2.clone()  # Feed Forward 기울기 복사
        grad_x_norm = grad_norm_2.clone()  # Residual 연결에서 입력 x 기울기 복사

        # Feed Forward 네트워크의 역전파
        grad_feed_forward = self.feed_forward.backward(grad_feed_forward)

        # Residual 연결에서 기울기 추가
        grad_x_norm += grad_feed_forward

        # 첫 번째 LayerNorm의 역전파
        grad_norm_1 = self.layer_norm_1.backward(grad_x_norm)

        # 어텐션 출력과 Residual 연결의 기울기 처리
        grad_attention = grad_norm_1.clone()  # 어텐션 기울기 복사
        grad_x = grad_norm_1.clone()  # Residual 연결에서 입력 x 기울기 복사

        # 멀티헤드 어텐션 블록의 역전파
        grad_attention = self.attention.backward(grad_attention)

        # Residual 연결에서 기울기 추가
        grad_x += grad_attention

        return grad_x  # 입력 기울기 반환


class GPT:
    def __init__(self, 
                 vocab_size: int, 
                 embed_size: int, 
                 max_seq_len: int, 
                 heads: int, 
                 ff_dim: int, 
                 num_blocks: int, 
                 lr: float = 1e-3):
        # 초기 설정
        self.lr: float = lr  # 학습률
        self.vocab_size: int = vocab_size  # 어휘 크기
        self.heads: int = heads  # 멀티헤드 어텐션의 헤드 수
        self.ff_dim: int = ff_dim  # 피드포워드 레이어의 내부 차원
        self.embed_size: int = embed_size  # 임베딩 차원
        self.max_seq_len: int = max_seq_len  # 최대 시퀀스 길이
        self.num_blocks: int = num_blocks  # 트랜스포머 블록 수

        # 토큰 임베딩 및 위치 임베딩 초기화
        self.token_embedding: Embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(max_seq_len, embed_size)

        # 트랜스포머 블록 생성
        self.transformer_blocks: List[TransformerEncoderBlock] = []
        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerEncoderBlock(embed_size, heads, ff_dim))

        # 출력 프로젝션 초기화
        self.output: OutputProjection = OutputProjection(embed_size, vocab_size)

        # 학습 모드 플래그
        self.train_mode: bool = True

        # 매개변수 및 기울기 관리
        self.param_and_grads: Dict = {
            "embedding_weight": self.token_embedding.weights,
            "embedding_weight_grad": self.token_embedding.grad_weights,
            "transformer_block": [
                {
                    "attention": {
                        "W_Q": block.attention.attention.W_Q,
                        "W_K": block.attention.attention.W_K,
                        "W_V": block.attention.attention.W_V,
                        "W_O": block.attention.attention.W_O,
                        "W_Q_grad": block.attention.attention.grad_W_Q,
                        "W_K_grad": block.attention.attention.grad_W_K,
                        "W_V_grad": block.attention.attention.grad_W_V,
                        "W_O_grad": block.attention.attention.grad_W_O,
                    },
                    "layernorm_1": {
                        "gamma": block.layer_norm_1.gamma,
                        "beta": block.layer_norm_1.beta,
                        "gamma_grad": block.layer_norm_1.grad_gamma,
                        "beta_grad": block.layer_norm_1.grad_beta,
                    },
                    "FeedForward": {
                        "fc1_weights": block.feed_forward.fc1.weights,
                        "fc1_bias": block.feed_forward.fc1.bias,
                        "fc1_weights_grad": block.feed_forward.fc1.grad_weights,
                        "fc1_bias_grad": block.feed_forward.fc1.grad_bias,
                        "fc2_weights": block.feed_forward.fc2.weights,
                        "fc2_bias": block.feed_forward.fc2.bias,
                        "fc2_weights_grad": block.feed_forward.fc2.grad_weights,
                        "fc2_bias_grad": block.feed_forward.fc2.grad_bias,
                    },
                    "layernorm_2": {
                        "gamma": block.layer_norm_2.gamma,
                        "beta": block.layer_norm_2.beta,
                        "gamma_grad": block.layer_norm_2.grad_gamma,
                        "beta_grad": block.layer_norm_2.grad_beta,
                    }
                } for block in self.transformer_blocks
            ],
            "output_weight": self.output.W,
            "output_weight_grad": self.output.grad_W,
        }

        # 매개변수 및 기울기 준비
        param_dict = {
            "embedding_weight": self.token_embedding.weights,
            "embedding_weight_grad": self.token_embedding.grad_weights,
            "output_weight": self.output.W,
            "output_weight_grad": self.output.grad_W
        }

        # 트랜스포머 블록 매개변수 추가
        for i, block in enumerate(self.transformer_blocks):
            # 멀티헤드 어텐션 파라미터
            for head_idx in range(block.attention.attention.heads):
                param_dict[f"transformer_block_{i}_attention_W_Q_{head_idx}"] = block.attention.attention.W_Q[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_Q_{head_idx}_grad"] = block.attention.attention.grad_W_Q[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_K_{head_idx}"] = block.attention.attention.W_K[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_K_{head_idx}_grad"] = block.attention.attention.grad_W_K[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_V_{head_idx}"] = block.attention.attention.W_V[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_V_{head_idx}_grad"] = block.attention.attention.grad_W_V[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_O_{head_idx}"] = block.attention.attention.W_O[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_O_{head_idx}_grad"] = block.attention.attention.grad_W_O[head_idx]

            # 어텐션 레이어 정규화 파라미터
            param_dict[f"transformer_block_{i}_attention_layernorm_gamma"] = block.attention.layer_norm.gamma
            param_dict[f"transformer_block_{i}_attention_layernorm_gamma_grad"] = block.attention.layer_norm.grad_gamma
            param_dict[f"transformer_block_{i}_attention_layernorm_beta"] = block.attention.layer_norm.beta
            param_dict[f"transformer_block_{i}_attention_layernorm_beta_grad"] = block.attention.layer_norm.grad_beta

            # Layer Norm 1 파라미터
            param_dict[f"transformer_block_{i}_layernorm_1_gamma"] = block.layer_norm_1.gamma
            param_dict[f"transformer_block_{i}_layernorm_1_gamma_grad"] = block.layer_norm_1.grad_gamma
            param_dict[f"transformer_block_{i}_layernorm_1_beta"] = block.layer_norm_1.beta
            param_dict[f"transformer_block_{i}_layernorm_1_beta_grad"] = block.layer_norm_1.grad_beta

            # Layer Norm 2 파라미터
            param_dict[f"transformer_block_{i}_layernorm_2_gamma"] = block.layer_norm_2.gamma
            param_dict[f"transformer_block_{i}_layernorm_2_gamma_grad"] = block.layer_norm_2.grad_gamma
            param_dict[f"transformer_block_{i}_layernorm_2_beta"] = block.layer_norm_2.beta
            param_dict[f"transformer_block_{i}_layernorm_2_beta_grad"] = block.layer_norm_2.grad_beta

            # FeedForward 파라미터
            param_dict[f"transformer_block_{i}_fc1_weights"] = block.feed_forward.fc1.weights
            param_dict[f"transformer_block_{i}_fc1_weights_grad"] = block.feed_forward.fc1.grad_weights
            param_dict[f"transformer_block_{i}_fc1_bias"] = block.feed_forward.fc1.bias
            param_dict[f"transformer_block_{i}_fc1_bias_grad"] = block.feed_forward.fc1.grad_bias

            param_dict[f"transformer_block_{i}_fc2_weights"] = block.feed_forward.fc2.weights
            param_dict[f"transformer_block_{i}_fc2_weights_grad"] = block.feed_forward.fc2.grad_weights
            param_dict[f"transformer_block_{i}_fc2_bias"] = block.feed_forward.fc2.bias
            param_dict[f"transformer_block_{i}_fc2_bias_grad"] = block.feed_forward.fc2.grad_bias

        # 옵티마이저를 위한 매개변수와 기울기 분리
        params = {}
        grads = {}
        for k, v in param_dict.items():
            if "_grad" in k:
                grads[k.replace("_grad", "")] = v
            else:
                params[k] = v

        # Adam 옵티마이저 초기화
        self.optimizer = AdamOptimizer(params=params, grads=grads, lr=lr)

    def forward(self, x: Tensor, temperature: float = 1.0) -> Tensor:
        # 순전파
        self.input_indices = x  # 입력 저장
        x = self.token_embedding.forward(x)  # 토큰 임베딩
        x = self.positional_encoding.forward(x)  # 위치 임베딩 추가
        for block in self.transformer_blocks:
            x = block.forward(x)  # 각 트랜스포머 블록을 통과
        logits = self.output.forward(x)  # 출력 로짓 계산
        self.logits = logits / temperature  # 온도 스케일링 적용
        probs = torch.softmax(self.logits, dim=-1)  # # 텐서의 끝 축의 확률 계산 
        return probs

    def backward(self, probs: Tensor, labels: Tensor) -> None:
        # 역전파 수행
        batch_size, vocab_size = probs.shape
        labels_one_hot = torch.zeros_like(probs)
        labels_one_hot[range(batch_size), labels] = 1

        # 로짓에 대한 손실 기울기 계산
        grad_logits = (probs - labels_one_hot) / batch_size

        # 출력 프로젝션에서 역전파
        grad_output = self.output.backward(grad_logits)

        # 트랜스포머 블록을 역순으로 역전파
        for block in reversed(self.transformer_blocks):
            grad_output = block.backward(grad_output)

        # 토큰 임베딩에서 역전파
        self.token_embedding.backward(grad_output)

    def update_parameters(self):
        # 매개변수 업데이트
        self.optimizer.step()

    def zero_grad(self):
        # 모든 기울기를 0으로 초기화
        self.token_embedding.grad_weights.zero_()
        self.output.grad_W.zero_()

        for block in self.transformer_blocks:
            # 어텐션 가중치의 기울기 초기화
            attention = block.attention.attention
            for i in range(attention.heads):
                attention.grad_W_Q[i].zero_()
                attention.grad_W_K[i].zero_()
                attention.grad_W_V[i].zero_()
                attention.grad_W_O[i].zero_()

            # 정규화 레이어의 기울기 초기화
            block.attention.layer_norm.grad_gamma.zero_()
            block.attention.layer_norm.grad_beta.zero_()
            block.layer_norm_1.grad_gamma.zero_()
            block.layer_norm_1.grad_beta.zero_()
            block.layer_norm_2.grad_gamma.zero_()
            block.layer_norm_2.grad_beta.zero_()

            # 피드포워드 레이어의 기울기 초기화
            block.feed_forward.fc1.grad_weights.zero_()
            block.feed_forward.fc1.grad_bias.zero_()
            block.feed_forward.fc2.grad_weights.zero_()
            block.feed_forward.fc2.grad_bias.zero_()

    def graceful_exit(self, signum, frame):
        # 훈련 중단 시 호출되어 모델과 설정을 저장하는 메서드

        import json  # JSON 형식으로 데이터를 저장하거나 로드하기 위한 라이브러리
        import os  # 파일 작업(경로 확인, 파일 읽기/쓰기 등)을 위한 라이브러리

        # 종료 신호를 감지하고 알림 메시지 출력
        print(f"Signal {signum} detected. Exiting gracefully.")

        # 모델 저장
        self.save_model(self.model_path)  # 지정된 경로에 모델 저장
        print(f"Model saved to {self.model_path}")  # 모델 저장 경로 출력

        # 설정 파일이 이미 존재하는지 확인
        if os.path.exists(self.config_path):
            # 설정 파일이 존재하면 로드
            with open(self.config_path, "r") as f:
                config = json.load(f)  # JSON 형식으로 로드
        else:
            # 설정 파일이 없으면 초기 설정 생성
            config = {
                "epoch": 0,  # 초기 에포크 값
                "loss": []   # 초기 손실 기록 리스트
            }

        # 현재 에포크를 기존 설정 값에 더함
        config["epoch"] += self.epoch

        # 현재까지의 손실 기록을 기존 손실 리스트에 추가
        config['loss'].extend(self.loss_history)

        # 업데이트된 설정을 파일에 저장
        with open(self.config_path, "w") as f:
            json.dump(config, f)  # JSON 형식으로 설정 저장

        print(f"Training history saved to {self.config_path}")  # 설정 저장 경로 출력

        # 프로그램 종료
        sys.exit(0)


    def train_model(self, 
                    data: List[Tensor], 
                    epochs: int, 
                    **kwargs) -> List[float]:
        # 모델 훈련을 수행하는 메서드

        # 키워드 인수에서 모델 저장 경로를 가져옴
        self.model_path = kwargs.get("model_path")
        # 키워드 인수에서 설정 파일 저장 경로를 가져옴
        self.config_path = kwargs.get("config_path")

        # 종료 신호(SIGINT, SIGTERM)를 graceful_exit 메서드로 처리하도록 설정
        signal.signal(signal.SIGINT, self.graceful_exit)
        signal.signal(signal.SIGTERM, self.graceful_exit)

        # 손실 기록을 저장할 리스트 초기화
        self.loss_history = []
        # 현재 에포크를 저장할 변수 초기화
        self.epoch = 0

        # 에포크 루프 (진행 상황 표시를 위해 tqdm 사용)
        for epoch in tqdm(range(epochs)):
            self.epoch = epoch  # 현재 에포크 값을 업데이트
            epoch_loss = 0.0  # 현재 에포크의 누적 손실 초기화

            # 데이터셋 내 배치 루프
            for input_indices in data:
                # 입력과 라벨 분리 (다음 토큰 예측 문제)
                labels = input_indices[1:]      # 라벨: 입력 시퀀스의 다음 토큰
                input_indices = input_indices[:-1]  # 입력: 마지막 토큰을 제외한 부분

                # 순전파 수행: 입력에 대한 확률 분포 계산
                probs = self.forward(input_indices)

                # 라벨을 원-핫 인코딩으로 변환
                labels_one_hot = torch.zeros_like(probs)  # 확률 분포 크기와 동일한 텐서 생성
                labels_one_hot[range(len(labels)), labels] = 1.0  # 라벨 위치에 1 할당

                # 작은 값을 더해 log(0) 방지
                eps = 1e-10
                # 교차 엔트로피 손실 계산
                loss = -(labels_one_hot * torch.log(probs + eps)).sum(dim=-1).mean()

                # 현재 배치의 손실 값을 에포크 손실에 더함
                epoch_loss += loss.item()

                # 역전파 수행: 손실에 대한 기울기 계산
                self.backward(probs, labels)

                # 매개변수 업데이트
                self.update_parameters()

                # 기울기를 초기화하여 이전 배치의 기울기가 누적되지 않도록 함
                self.zero_grad()

            # 에포크 평균 손실 계산
            avg_loss = epoch_loss / len(data)  # 전체 배치 손실의 평균값 계산
            self.loss_history.append(avg_loss)  # 에포크 손실을 기록 리스트에 추가
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")  # 현재 에포크 결과 출력

            # 손실 값이 NaN이면 학습 중단
            if torch.isnan(torch.tensor(avg_loss)):
                print("Loss became NaN. Stopping training.")  # NaN 감지 시 종료 메시지 출력
                break

        # 전체 학습 손실 기록 반환
        return self.loss_history  # 모든 에포크의 손실 기록을 반환

    def eval_mode(self):
        # 평가 모드 설정: 학습 관련 연산 비활성화
        self.train_mode = False   

    def save_model(self, path):
        # 모델 저장 (PyTorch의 torch.save 사용)
        torch.save(self, path)

    @staticmethod
    def load_model(path):
        # 저장된 모델 불러오기
        return torch.load(path)

    def generate_sequence(self, initial_input, max_length):
        # 새로운 시퀀스 생성 메서드

        # 평가 모드로 전환
        self.eval_mode()

        # 초기 입력을 복사하여 수정 가능한 텐서로 만듦
        input_indices = initial_input.clone()

        # 최대 길이까지 반복하여 시퀀스 생성
        for _ in range(max_length - len(initial_input)):
            # 순전파 수행: 현재 입력에 대한 확률 계산
            probs = self.forward(input_indices)

            # 마지막 토큰의 확률 분포 가져오기
            next_token_probs = probs[-1]

            # 확률 분포에서 다음 토큰 선택 (argmax를 사용하여 가장 높은 확률 선택)
            next_token = torch.argmax(next_token_probs)

            # 선택한 토큰을 입력 시퀀스에 추가
            input_indices = torch.cat((input_indices, next_token.unsqueeze(0)), dim=0)

            # 입력 시퀀스 길이가 최대 길이를 초과하면 뒤에서 자르기
            if len(input_indices) > self.max_seq_len:
                input_indices = input_indices[-self.max_seq_len:]

        # 생성된 시퀀스 반환
        return input_indices
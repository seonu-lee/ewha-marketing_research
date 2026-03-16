'''
파이선 기본 문법 리뷰
print

number
string 
list
tuple
dictionary
set

if
for
while
function
lamda function
package
regular expression
time

'''

#===================================
# print
#===================================
print('marketing start!')
print(123)

name = 'lee'
print(f'{name}, Hello')

#===================================
# 변수
#===================================
### 변수 선언
str_var = "marketing" # 문자열 (string)
num_int = 123 # 숫자 - 정수(integer)
num_float = 123.54 # 숫자 - 실수(float)
bool_var = True # boolean(논리형) True/False 
print(str_var, num_int, num_float, bool_var)

### 변수간 연산
a = 2
b = 3
a, b = 2, 3 
a=2; b=3

a+b
a*b 
a/b

a>b
a<=b
a==b
a!=b

### 타입, id 확인
type(a)
type(123.0)
id(a) # 고유id 확인

### 허용되는 변수명 선언 방법
price = 100
_price = 100
#1price = 100 # 변수명은 숫자로 시작할 수 없음
print1 = 100
변수 = 100

### 예약어는 변수명 사용 불가능
# False = 100
# if = 100 
IF = 100 # 대소문자 구분하기 때문에 IF는 예약어가 아님

#===================================
# number
#===================================
### 연산
print(1+2)
print(2*3)
print(2**3)
print(7/3)

print(7//3) # 몫
print(7%3) # 나머지
abs(-2)

# 외부 모듈 사용
import math
math.pi
math.ceil(2.52) # x 이상의 수 중에서 가장 작은 정수

#===================================
# string
#===================================
### 문자열 생성
str1 = "marketing"
str1 = 'marketing'
str1 = """marketing"""
str1 = '''marketing'''

### 이스케이프 문자
# 문자열 내 특수한 의미를 가지는 약속된 문자

# 문자열내 ", ' 표현 방법
# str1 = "Do you have a "book"?" # 문자열 내에서 "가 문자열의 시작과 끝을 구분하는 역할을 하기 때문에 오류 발생
str2 = "Do you have a \"book\"?" # 따옴표 자체 (\", \')
print(str2)

str3 = 'Do you have a "book"?' 
str4 = '''Do you have a "book"?'''
print(str3, str4)

# 문자열 내 탭/줄바꿈/백슬래시 표현 방법
a1 = "marketing \t class"
a2 = "begins \n ended"
print(a1) 
print(a2)

# marketing 	 class
# begins 
#  ended

# 문자열 내 역슬래시 표현 방법
a3 = "markeitng\accounting"
a4 = "markeitng\\accounting"
print(a3)
print(a4)

# raw string literal 
# 이스케이프 문자로 해석되지 않고 그대로 표현
a5 = r"marketing \t class"
a6 = "marketing \\t class" # 동일한 결과
print(a5)
print(a6)
# marketing \t class
# marketing \t class

### 멀티라인 입력 
multi_line_str = """
string
multi line
test
"""
print(multi_line_str)
# string
# multi line
# test

multi_line_str2 = """string
multi line
test"""
print(multi_line_str2)
# string
# multi line
# test

multi_line_str3 = "string\nmulti line\ntest" # 줄바꿈을 이용해 멀티라인 표현
print(multi_line_str3)
# string
# multi line
# test

### 문자열 연산
str1 = 'apple'
str2 = 'banana'
str3 = 'seoul#daejeon#jinju'

print(str1*3)
print(str1 + str2)
print('a' in str1) # string은 시퀀스이기 때문에 문자 하나하나를 리스트로 인식하여 in연산자 사용가능
print('i' in str1)
print('P' not in str2)

# 문자열 함수
print(str1.upper()) # 대문자로 변환
print(str1.lower()) # 소문자로 변환
print(str1.capitalize()) #시작 글자를 대문자로

print(str2.endswith("a")) #특정 문자로 끝나는지 확인
print(str2.startswith("a")) #문장이 특정 문자로 시작하는지 확인
print(str1.replace("ple", "thanks"))
print(sorted(str1)) #정열해서 리스트형태로 반환
print(str3.split('#')) #지정한 문자를 기준으로 분리하여 리스트로 반환, 아무것도 지정하지 않으면 스페이스를 기준으로 분리함.

str4= "apple ate  banana "
print(str4.replace(" ",""))
      
#===================================
# list
#===================================
# 리스트 자료형 (순서o, 중복o, 수정o, 삭제o)

### 선언
a1 = [] # 대괄호로 빈 리스트 선언, a1은 list타입의 빈 리스트가 됨
a2 = list() # list()는 빈 리스트를 만드는 함수, list()안에 다른 컬렉션을 넣으면 리스트로 변환해줌
a3 = [51, 52, 53, 54, 55] # 리스트는 대괄호로 표현, 원소는 쉼표로 구분, 원소의 타입은 다양할 수 있음
a4 = [5, 6, True, 'marketing', 'data analysis'] # 리스트는 다양한 타입의 원소를 담을 수 있음
a5 = [5, 6, ['marketing', 'data'], True] # 리스트 안에 리스트도 가능, a5[2]는 ['marketing', 'data']라는 리스트가 됨

### 추출
a3[0] # 시작 인덱스는 0임 #파이썬에서는 모든 인덱스가 0부터 시작임 #R에서는 1부터 시작
a3[4]
a3[-1] # 맨 마지막 값
a3[0:3] # 인덱스 범위
a4[-1][:3] # a4[-1]은 'data analysis'라는 문자열이 되고, 문자열도 시퀀스이기 때문에 인덱싱과 슬라이싱 가능 #[:3]은 인덱스 0,1,2에 해당하는 문자만 추출하라는 의미
a4[-1][2:] #[2:]는 인덱스 2부터 끝까지 추출하라는 의미

### 추가
b1 = ['math','language','physics']
b1.append('coding') # 리스트의 맨 뒤에 원소 추가
print(b1)

### 삭제
b1.remove('math') # 지정한 값을 제거
print(b1)
del b1[-1] # 인덱스로 지정하여 삭제 # del은 리스트에서 원소를 제거하는 명령어, del b1[-1]은 b1의 맨 마지막 원소를 제거하라는 의미
print(b1)

### 수정
b1[0] = "new element"

### 정렬
b2 = ['math','language','physics']
b2_sorted = sorted(b2) # 오름차순 정렬
b2_sorted_r = sorted(b2, reverse=True) # 내림차순 정렬

### 리스트 원소 합치기
b2 = ['math','language','physics']
" ".join(b2)


#===================================
# tuple
#===================================
# 튜플 자료형 (순서o, 중복o, 수정x, 삭제x), 불변 (한번 선언해서 끝까지 사용)

### 선언
a1 = () # 빈 튜플 선언, a1은 tuple타입의 빈 튜플이 됨
a2 = (111,) #원소가 하나일때는 ,를 찍어야 튜플로 인식 #괄호가 있어도 원소가 하나면 튜플이 아님
e1 = (111) #괄호가 있어도 원소가 하나면 튜플이 아님, e1은 int타입의 숫자 111이 됨
print(type(a1), type(a2), type(e1))

a3 = (1, 11, ('math', 'language', 'physics')) # 튜플은 소괄호로 표현, 원소는 쉼표로 구분, 원소의 타입은 다양할 수 있음
a4 = "a", "b" # 괄호가 없어도 튜플 # a4는 튜플이 됨
print(type(a3), type(a4))

### 인덱싱
a3[1] # 인덱스 1에 해당하는 원소는 11임
a3[-1] # a3[-1]은 a3의 맨 마지막 원소인 ('math', 'language', 'physics')라는 튜플이 됨
a3[-1][0] # a3[-1]은 ('math', 'language', 'physics')라는 튜플이 되고, 튜플도 시퀀스이기 때문에 인덱싱과 슬라이싱 가능 # a3[-1][0]은 'math'라는 문자열이 됨
a3[-1][0][0] # a3[-1][0]은 'math'라는 문자열이 되고, 문자열도 시퀀스이기 때문에 인덱싱과 슬라이싱 가능 # a3[-1][0][0]은 'm'이라는 문자 하나가 됨

### 연산
a3+a4
a4*2

### 수정(불가)
a4[0] = "add new" # 튜플은 불변이기 때문에 a4[0] = "add new"는 a4의 원소를 수정하려는 시도이므로 오류 발생

### 변수값 swapping
x,y = 11,22 
x,y = y,x # x,y = y,x는 x와 y의 값을 서로 바꾸라는 의미, x는 22가 되고, y는 11이 됨
print(x,y)

### list <-> tuple 변환
list((1,2)) # tuple을 list로 변환
tuple([1,2]) # list를 tuple로 변환

### 언팩킹
c1 = 1, 2, 3 # c1은 튜플이 됨, c1 = (1, 2, 3)과 동일한 의미
d1, d2, d3 = c1 # c1은 튜플이 되고, d1, d2, d3는 c1의 원소를 각각 받는 변수들이 됨, d1은 1이 되고, d2는 2가 되고, d3는 3이 됨
e1, e2, e3 = 1, 2, 3 # e1, e2, e3는 1, 2, 3을 각각 받는 변수들이 됨, e1은 1이 되고, e2는 2가 되고, e3는 3이 됨


#===================================
# dictionary
#===================================
# 딕셔너리 자료형 (순서x, 중복x, 수정o, 삭제o)

### 선언
a1 = {}
a2 = dict()

a3 = {
  'name': 'lee',
  'score': 95,
  'courses': ['math', 'science', 'coding'],
  'enrolled': True
} # 딕셔너리는 중괄호로 표현, key와 value는 :로 구분, key-value 쌍은 쉼표로 구분, key는 고유해야하지만 value는 중복될 수 있음

a4 = dict(
  name = 'lee',
  score = 95,
  courses = ['math', 'science', 'coding'],
  enrolled = True
)

### 입출력, 수정
a4['name'] # 출력 - name이라는 key가 없으면 에러 발생시킴
a4.get('name', 0) # 출력 - name이라는 key가 없으면 별도로 지정한 값 반환(0)
a4.get('ame', 0) # ame이라는 key가 없으면 0 반환
a4.get('ame') # ame이라는 key가 없으면 None 반환
a4['birth'] = '2010-01-25' # 추가 - birth이라는 key가 없으면 새로 추가되고, 있으면 기존 value가 수정됨
a4['score'] = 50 # 수정
del a4['birth'] # 삭제

### 활용
len(a4) # 딕셔너리의 원소 개수는 key의 개수와 동일함, a4에는 name, score, courses, enrolled라는 4개의 key가 있으므로 len(a4)는 4가 됨
a4.keys() # 딕셔너리의 key를 리스트 형태로 반환, a4에는 name, score, courses, enrolled라는 4개의 key가 있으므로 a4.keys()는 ['name', 'score', 'courses', 'enrolled']라는 리스트가 됨
a4.values() # 딕셔너리의 value를 리스트 형태로 반환, a4에는 name, score, courses, enrolled라는 4개의 key가 있고, 각각의 value는 'lee', 50, ['math', 'science', 'coding'], True이므로 a4.values()는 ['lee', 50, ['math', 'science', 'coding'], True]라는 리스트가 됨
a4.items() # (key, value) 튜플형태로 리스트에 담음 


#===================================
# set
#===================================
# 집합(set) 자료형 (순서x, 중복x, 수정x, 삭제x) 
# 집합은 수학에서의 집합과 유사한 개념으로, 순서가 없고, 중복이 허용되지 않는 데이터 구조임. 집합은 수정과 삭제가 불가능한 불변(immutable) 자료형이지만, 새로운 원소를 추가하는 것은 가능함.

### 선언
a1 = set()
a2 = {'a', 'b', 'c', 'd'} #키값 없이 {}사용하면 집합
a3 = [1, 2, 3, 1, 2, "a"] # a3는 중복이 있는 리스트이지만, set(a3)를 하면 a3의 원소 중에서 중복이 제거된 집합이 됨, a3에는 1, 2가 각각 2개씩 중복되어 있지만, set(a3)는 1, 2가 각각 1개씩만 포함된 집합이 됨
a4 = set(a3) # list->set, 중복제거됨 # {1, 2, 3, 'a'}

### 연산
a2 | a4 # 합집합 # a2와 a4의 원소를 모두 포함하는 집합
a2 - a4 # 차집합 # a2에는 'a', 'b', 'c', 'd'가 있고, a4에는 1, 2, 3, 'a'가 있으므로 a2 - a4는 a2의 원소 중에서 a4에 없는 원소인 'b', 'c', 'd'가 포함된 집합이 됨
a2 & a4 # 교집합 # a2와 a4의 원소 중에서 공통된 원소인 'a'가 포함된 집합이 됨
a2 ^ a4 # 대칭차집합  # a2와 a4의 원소 중에서 서로 다른 원소인 1, 2, 3, 'b', 'c

### 리스트의 중복제거
a5 = list(set(a3)) # set->list # 중복을 제거한 리스트를 만드는 방법 


#===================================
# if (조건문)
#===================================
### 논리연산자 (boolean) 
a, b, c = 10, 20, 30

a<b and b>c 
a<b or b>c
not a>b

x1 = [1, 2, 3] # 리스트는 순서가 있고, 중복이 허용되는 자료형이므로 x1은 리스트가 됨, x1에는 1, 2, 3이라는 원소가 순서대로 포함되어 있음
x2 = {7, 8, 9, 10} # 집합은 순서가 없고, 중복이 허용되지 않는 자료형이므로 x2는 집합이 됨, x2에는 7, 8, 9, 10이라는 원소가 포함되어 있지만, 순서는 정해져 있지 않음
x3 = (5, 6, 7) # 튜플은 순서가 있고, 중복이 허용되는 자료형이지만, 수정과 삭제가 불가능한 불변(immutable) 자료형이므로 x3는 튜플이 됨, x3에는 5, 6, 7이라는 원소가 순서대로 포함되어 있지만, 원소를 수정하거나 삭제할 수는 없음
x4 = {'name' : 'Lee', 'city' : "jinju", 'grade' : 'B'} # 딕셔너리는 순서가 없고, 중복이 허용되지 않는 자료형이지만, 수정과 삭제가 가능한 가변(mutable) 자료형이므로 x4는 딕셔너리가 됨, x4에는 'name', 'city', 'grade'라는 key와 각각의 value인 'Lee', 'jinju', 'B'가 포함되어 있지만, 순서는 정해져 있지 않음

20 in x1 # x1에 20이 포함되어 있는지 확인
90 in x2 # x2에 90이 포함되어 있는지 확인
12 not in x3 # x3에 12이 포함되어 있지 않은지 확인
'name' in x4 # x4에 'name'이라는 key가 있는지 확인
'name' in x4.keys() # x4의 key 중에 'name'이 있는지 확인
'seoul' in x4.values() # x4의 value 중에 'seoul'이 있는지 확인

### 조건문
score, grade = 85, 'A+'
if score >=80 and grade == 'A+':
  print('Pass')
else:
  print('Fail')

name = 'lee'
invited = ['lee', 'kim', 'park']
if name in invited:
  print(f"{name}, you are in the list")

### 다중조건문
score = 85
if score >= 90:
  print('outcome : A')
elif score >= 80:
  print('outcome : B')
elif score >= 70:
  print('outcome : C')
else:
  print('FAIL')


#===================================
# for (반복문)
#===================================
total = 0
for v in range(1, 11):
    total += v # range(1, 11)은 1부터 10까지의 숫자를 생성하는 함수이므로, for v in range(1, 11)은 v가 1부터 10까지의 숫자를 순서대로 받는 반복문이 됨, total += v는 total = total + v와 동일한 의미로, total에 v를 더해서 다시 total에 저장하라는 의미이므로, total은 1부터 10까지의 숫자의 합인 55가 됨
print(total)
print(sum(range(1,11))) # 동일한 결과 # range(1, 11)은 1부터 10까지의 숫자를 생성하는 함수이므로, sum(range(1, 11))은 1부터 10까지의 숫자의 합인 55가 됨

### for문 이용한 중복제거
a1 = [9, 1, 2, 8, 4, 5, 2, 1, 3, 4, 4]
a1_new = list()
for item in a1:
    if item not in a1_new:
        a1_new.append(item) # a1_new에 item이 없으면 a1_new에 item을 추가하라는 의미이므로, a1_new에는 a1의 원소 중에서 중복이 제거된 원소들이 순서대로 포함되어 있음
print(sorted(a1_new)) # sorted(a1_new)는 a1_new의 원소를 오름차순으로 정렬한 리스트를 반환하므로, sorted(a1_new)는 a1의 원소 중에서 중복이 제거되고 오름차순으로 정렬된 리스트가 됨
print(set(a1)) # 동일한 결과

### 사전에 for문 적용
a2 = dict(
  name = 'lee',
  score = 95,
  courses = ['math', 'science', 'coding'],
  enrolled = True
)
for key, value in a2.items():
    print(f"- key: {key}, value: {value}") # a2.items()는 (key, value) 튜플형태로 리스트에 담은 것을 반환하므로, for key, value in a2.items()는 a2의 key와 value를 각각 key와 value라는 변수에 받는 반복문이 됨, print(f"- key: {key}, value: {value}")는 key와 value를 출력하는 명령어이므로, a2의 key와 value가 각각 출력됨

### break - 반복문에서 빠져나가기
number_list = [1, 22, 14, 17, 100, 204, 117, 25, 340, 18]
for number in number_list:
  if number == 17:
    print(f"{number} is in the list!!")
    break # 반복문에서 빠져나가라는 의미이므로, number_list에서 17을 찾으면 "17 is in the list!!"가 출력되고, 반복문이 종료됨
  else:
    print(f"{number} not in the list")

### continue - 이하를 실행하지 않고 반복문 계속
scores = [1, 2, 3, "name", 4, True, 5]
total = 0
for value in scores:
    if type(value) in [bool, str]:        
        print('type error, skip adding:', value)
        continue # value의 타입이 bool이나 str이면 "type error, skip adding: value"가 출력되고, continue는 이하를 실행하지 않고 반복문을 계속하라는 의미이므로, total += value는 실행되지 않고, 다음 value로 넘어가게 됨
    total += value # 여기까지 하면 total은 저장만되고 출력안됨 
print(total) # scores에는 1, 2, 3, "name", 4, True, 5가 포함되어 있지만, "name"은 str타입이고, True는 bool타입이므로, total에는 1, 2, 3, 4, 5가 더해지게 되고, total은 15가 됨
    
### enumerate - 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
str_list = ['a', 'b', 'c', 'd', 'e']
for each in enumerate(str_list):
  print(each)

for i,v in enumerate(str_list):
  print("index: {}, value: {}".format(i,v))
  # print(f"index: {i}, value: {v}") # 동일한 결과

### list내에서 for문
[i*3 for i in range(5)] # list내에서 for문을 사용할 때는 대괄호로 감싸야 함, i*3 for i in range(5)는 i가 0부터 4까지의 숫자를 순서대로 받는 반복문이면서, i*3은 i에 3을 곱한 값을 의미하므로, [i*3 for i in range(5)]는 0부터 4까지의 숫자에 각각 3을 곱한 값을 순서대로 포함하는 리스트가 됨

new_v = list()
for i in range(5) :
  new_v.append(i*3)
print(new_v) # 동일한 결과

#===================================
# while
#===================================
# 조건이 만족하는 한 계속 반복
n = 10
while n > 0:
  print(n)
  n -= 1
  
  
#===================================
# function
#===================================
# 함수1
def func_mul(num):
    new_num = num*2 # new_num은 이 함수내에서만 유효한 지역변수
    return new_num # return은 함수의 결과값을 반환하는 명령어, func_mul(10)은 num에 10을 넣어서 func_mul 함수를 실행하라는 의미이므로, func_mul(10)은 new_num = 10*2가 되고, new_num은 20이 되며, return new_num은 20을 반환하라는 의미이므로, func_mul(10)은 20이 됨

a = func_mul(10)
print(a)
# new_num # 위 함수내에서만 유효한 지역변수여서 오류발생 # new_num은 func_mul 함수내에서만 유효한 지역변수이므로, func_mul 함수 밖에서는 new_num을 사용할 수 없어서 오류가 발생함

# 함수1
scale = 3 # 전역변수 - 함수내에서도 사용가능, 함수 밖에서도 사용가능
def func_mul2(num):
    new_num = num*scale # 함수내에서도 전역변수 사용가능
    return new_num

b = func_mul2(10)
print(b)

#===================================
# lambda 함수
#===================================
# lambda 인풋 값: return 값 # lambda 함수는 간단한 함수를 한 줄로 표현하는 방법으로, lambda 키워드 다음에 인풋 값을 받고, 콜론(:) 다음에 return 값을 표현하는 형태로 작성됨. lambda 함수는 이름이 없는 익명 함수이므로, 변수에 할당해서 사용할 수 있음
a = lambda x,y: x*y
a(5,6)

# 일반 함수
def mul_func(x, y):
  return x * y


#===================================
# package
#===================================
# /mylib 폴더에 패키지 파일을 생성하고, myfunctions 함수 작성 후 실행할것

### 패키지 불러오기
import sys
sys.path # 현재 등록된 path 확인
sys.path.append(r'C:\Users\seonu\Documents\ewha-marketing_research\session2_python_basic') # 백슬래시 때문에 오류남 /로바꾸거나 r붙이기 # 패키지가 있는 폴더 경로를 sys.path에 추가하여 패키지를 불러올 수 있도록 함
from mylib import myfunctions

result = myfunctions.add_nums(1,2)
result2 = myfunctions.multiply_nums(5,2)

### 업데이트 - myfunctions.exp_nums 활성화 후 실행할 것
result3 = myfunctions.exp_nums(5,2) # 오류

from mylib import myfunctions # 캐시에 저장된 것 불러옴. 수정사항 반영안됨
result3 = myfunctions.exp_nums(5,3) # 여전히 오류

from importlib import reload
reload(myfunctions) # 다시 불러오기
result3 = myfunctions.exp_nums(5,3)

print(__name__) # 직접 실행된 경우(해당 파일 내에서), __name__은 __main__ 이 됨
print(myfunctions.__name__) # import된 경우, __name__은 import된 파일이름이 됨


#===================================
# 정규표현식 (Regular Expression)
#===================================
# 문자열에서 특정 패턴을 찾거나 치환하기 위한 규칙 언어
import re

### match 함수: 문자열 처음부터 정규식과 매칭되는 패턴을 찾아서 리턴 
pattern = re.compile('[a-z]{1}') # 정규표현식 제작(compile)
print(pattern.match("1a2b3"))
print(pattern.match("a2b3"))

pattern.match("1a2b3") == None # 문자열의 처음부터 정규식과 매칭되는지 판정

### search 함수: 문자열 전체를 검색해서 정규식과 매칭되는 패턴을 찾아서 리턴
pattern.search("1a2b34") == None # 문자열 전체에서 해당 패턴이 있는지 판정

### findall 함수: 정규표현식과 매칭되는 모든 문자열을 리스트로 반환
sentence = "Applying Pythton37 to Marketing2 types of data"

pattern = re.compile('[a-zA-Z]+[0-9]{1}') # 알파벳(a–z, A–Z) 1개 이상(+) 연속되고 이어서 숫자 1개가 있는 부분을 찾는 정규표현식 컴파일
words = pattern.findall(sentence)
print(words)

### split 함수: 정규표현식 패턴 문자열을 기준으로 문자열을 분리하여 리스트로 반환
pattern = re.compile(':') # ':'을 찾는 정규표현식 컴파일
result = pattern.split('python:html:css') # 패턴를 기준으로 분자열 분리한 리스트 반환
result = re.split(':', 'python:html:css') # 위와 동일
print(result)

### sub 함수: 찾은 정규표현식 패턴 문자열을 다른 문자열로 변경
num = '880512-3456789'

pattern = re.compile('-') # '-'을 찾는 정규표현식 컴파일
pattern.sub('*', num)  # 정규표현식.sub(바꿀문자열, 대상문자열)
re.sub('-', '*', num) # sub(정규표현식, 바꿀문자열, 대상문자열) 
re.sub('-[0-9]{7}', '-*******', num) # sub(정규표현식, 바꿀문자열, 대상문자열) 

# 일반 패턴
'''
. : 줄바꿈 문자인 \n를 제외한 모든 문자 1개인 패턴
? : 문자가 0번 또는 1번 표시되는 패턴
* : 앞 문자가 0번 또는 그 이상 반복되는 패턴
+ : 앞 문자가 1번 또는 그 이상 반복되는 패턴
| : 여러개의 정규표현식을 한꺼번에 치환

{n} : 앞 문자가 n 번 반복되는 패턴
{m, n} : 앞 문자가 m 번 반복되는 패턴부터 n 번 반복되는 패턴까지

[abc] : a, b, c 중 1개인 패턴
[a-z] : 소문자 전체 중 1개인 패턴 
[a-zA-Z] : 대소문자 전체 중 1개인 패턴 
[a-zA-Z0-9] : 숫자대소문자 전체 중 1개인 패턴 

[^a-zA-Z0-9] : 숫자, 대소문자가 아닌 것. 즉, 모든 특수 문자
[^ \t\n\r\f\v] : 화이트 스페이스가 아닌 것 

[가-힣] : 모든 한글문자
[ㄱ-ㅎㅏ-ㅣ] : 한글 자모음
[a-zA-Z]+ : 소문자 또는 대문자가 하나 이상 포함된 패턴

# 자주 사용하는 정규식은 별도의 표기법으로 표현할 수 있음    
- \d - 숫자와 매치, [0-9]와 동일한 표현식
- \D - 숫자가 아닌 것과 매치, [^0-9]와 동일한 표현식
- \s - whitespace 문자와 매치, [ \t\n\r\f\v]와 동일한 표현식 맨 앞의 빈 칸은 공백문자(space)를 의미
- \S - whitespace 문자가 아닌 것과 매치, [^ \t\n\r\f\v]와 동일한 표현식
- \w - 문자+숫자(alphanumeric)와 매치
- \W - 문자+숫자(alphanumeric)가 아닌 문자와 매치

'''
# 참고
# 파이썬 문자열에서 "\"의 역할 - 줄바꿈, 탭, 백스페이스 등을 표시하게 위해 사용
text = "A B\tC\nD\rE\fF\vG"
print(text)
# 정규표현식에서도 "\"가 의미를 지님. 이를 위해서는 파이썬 문자열에서 "\"가 그대로 유지되어야함
text = r"A B\tC\nD\rE\fF\vG" # raw string literal - r""
print(text)


#===================================
# 시간
#===================================
import datetime

# 날짜/시간 객체 만들기
datetime.datetime(2018,5,19) # datetime.datetime(2018, 5, 19, 0, 0)
datetime.date(2018,5,19)  #datetime.date(2018, 5, 19)


# 문자열 --> 날짜/시간으로 변환 (strptime)
datetime.datetime.strptime("2018-5-12", "%Y-%m-%d")

# 날짜/시간 --> 문자열로 변환 (strftime)
datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') # 현재 날짜/시간을 '년-월-일-시-분-초' 형태의 문자열로 변환


# 특정 지역  시간
import pytz # pytz는 파이썬에서 시간대(timezone)를 다루기 위한 라이브러리로, 세계 각 지역의 시간대를 지원함. pytz를 사용하면 특정 지역의 현재 시간이나 날짜/시간을 쉽게 구할 수 있음
KST = pytz.timezone('Asia/Seoul')
datetime.datetime.now(KST).strftime('%Y-%m-%d-%H-%M-%S')
datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')

# 날짜/시간에 일정 날짜/시간 더하기 - timedelta
datetime.datetime.now() - datetime.timedelta(hours = 1) # 현재시간 1시간전의 시간 - pd.Timedelta(hours = 1)도 사용가능
datetime.datetime.now() + datetime.timedelta(days = 1) # 현재시간 1일후의 시간


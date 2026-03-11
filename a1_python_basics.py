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
str1 = "Do you have a "book"?" 
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

### 멀티라인 입력 
multi_line_str = """
string
multi line
test
"""
print(multi_line_str)

multi_line_str2 = "string\nmulti line\ntest" # 줄바꿈을 이용해 멀티라인 표현
print(multi_line_str2)

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
    
      
#===================================
# list
#===================================
# 리스트 자료형 (순서o, 중복o, 수정o, 삭제o)

### 선언
a1 = []
a2 = list()
a3 = [51, 52, 53, 54, 55]
a4 = [5, 6, True, 'marketing', 'data analysis']
a5 = [5, 6, ['marketing', 'data'], True]

### 추출
a3[0] # 시작 인덱스는 0임
a3[4]
a3[-1] # 맨 마지막 값
a3[0:3] # 인덱스 범위
a4[-1][:3]
a4[-1][2:]

### 추가
b1 = ['math','language','physics']
b1.append('coding')
print(b1)

### 삭제
b1.remove('math') # 지정한 갗을 제거
print(b1)
del b1[-1] # 인덱스로 지정하여 삭제
print(b1)

### 수정
b1[0] = "new element"

### 정렬
b2 = ['math','language','physics']
b2_sorted = sorted(b2) # 올림차순 정렬
b2_sorted_r = sorted(b2, reverse=True) # 내림차순 정렬

### 리스트 원소 합치기
b2 = ['math','language','physics']
"#".join(b2)


#===================================
# tuple
#===================================
# 튜플 자료형 (순서o, 중복o, 수정x, 삭제x), 불변 (한번 선언해서 끝까지 사용)

### 선언
a1 = ()
a2 = (111,) #원소가 하나일때는 ,를 찍어야 튜플로 인식
print(type(a1), type(a2))

a3 = (1, 11, ('math', 'language', 'physics'))
a4 = "a", "b" # 괄호가 없어도 튜플

### 인덱싱
a3[1]
a3[-1]

### 연산
a3+a4
a4*2

### 수정(불가)
a4[0] = "add new"

### 변수값 swapping
x,y = 11,22
x,y = y,x
print(x,y)

### list <-> tuple 변환
list((1,2))
tuple([1,2])

### 언팩킹
c1 = 1, 2, 3
d1, d2, d3 = c1
e1, e2, e3 = 1, 2, 3


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
}

a4 = dict(
  name = 'lee',
  score = 95,
  courses = ['math', 'science', 'coding'],
  enrolled = True
)

### 입출력, 수정
a4['name'] # 출력 - name이라는 key가 없으면 에러 발생시킴
a4.get('name', 0) # 출력 - name이라는 key가 없으면 별도로 지정한 값 반환(0)
a4['birth'] = '2010-01-25' # 추가
a4['score'] = 50 # 수정
del a4['birth'] # 삭제

### 활용
len(a4)
a4.keys()
a4.values()
a4.items() # (key, value) 튜플형태로 리스트에 담음


#===================================
# set
#===================================
# 집합(set) 자료형 (순서x, 중복x, 순서x)

### 선언
a1 = set()
a2 = {'a', 'b', 'c', 'd'} #키값 없이 {}사용하면 집합
a3 = [1, 2, 3, 1, 2, "a"]
a4 = set(a3) # list->set, 중복제거됨

### 연산
a2 | a4 # 합집합
a2 - a4 # 차집합

### 리스트의 중복제거
a5 = list(set(a3)) # set->list


#===================================
# if (조건문)
#===================================
### 논리연산자 (boolean)
a, b, c = 10, 20, 30

a<b and b>c
a<b or b>c
not a>b

x1 = [1, 2, 3]
x2 = {7, 8, 9, 10}
x3 = (5, 6, 7)
x4 = {'name' : 'Lee', 'city' : "jinju", 'grade' : 'B'}

20 in x1
90 in x2
12 not in x3
'name' in x4
'name' in x4.keys()
'seoul' in x4.values()

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
    total += v
sum(range(1,11))

### for문 이용한 중복제거
a1 = [9, 1, 2, 8, 4, 5, 2, 1, 3, 4, 4]
a1_new = list()
for item in a1:
    if item not in a1_new:
        a1_new.append(item)
print(a1_new)
print(set(a1)) # 동일한 결과

### 사전에 for문 적용
a2 = dict(
  name = 'lee',
  score = 95,
  courses = ['math', 'science', 'coding'],
  enrolled = True
)
for key, value in a2.items():
    print(f"- key: {key}, value: {value}")

### break - 반복문에서 빠져나가기
number_list = [1, 22, 14, 17, 100, 204, 117, 25, 340, 18]
for number in number_list:
  if number == 17:
    print(f"{number} is in the list!!")
    break
  else:
    print(f"{number} not in the list")

### continue - 이하를 실행하지 않고 반복문 계속
scores = [1, 2, 3, "name", 4, True, 5]
total = 0
for value in scores:
    if type(value) in [bool, str]:        
        print('type error, skip adding:', value)
        continue
    total += value
    
### enumerate - 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
str_list = ['a', 'b', 'c', 'd', 'e']
for each in enumerate(str_list):
  print(each)

for i,v in enumerate(str_list):
  print("index: {}, value: {}".format(i,v))
  # print(f"index: {i}, value: {v}") # 동일한 결과

### list내에서 for문
[i*3 for i in range(5)]


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
    return new_num

a = func_mul(10)
# new_num # 위 함수내에서만 유효한 지역변수여서 오류발생

# 함수1
scale = 3 # 전역변수
def func_mul2(num):
    new_num = num*scale # 함수내에서도 전역변수 사용가능
    return new_num
b = func_mul2(10)


#===================================
# lambda 함수
#===================================
# lambda 인풋 값: return 값
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
# sys.path # 현재 등록된 path 확인
sys.path.append('/Users/carrot/Dropbox/Learning/inflearn/902_textanalytics_class/class20261/s02_python_basics')
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
datetime.datetime.now().strftime('%Y-%m-%d')

# 특정 지역  시간
import pytz
KST = pytz.timezone('Asia/Seoul')
datetime.datetime.now(KST).strftime('%Y-%m-%d-%H-%M-%S')
datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')

# 날짜/시간에 일정 날짜/시간 더하기 - timedelta
datetime.datetime.now() - datetime.timedelta(hours = 1) # 현재시간 1시간전의 시간 - pd.Timedelta(hours = 1)도 사용가능
datetime.datetime.now() + datetime.timedelta(days = 1) # 현재시간 1일후의 시간


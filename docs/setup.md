# handout

# Miniconda 설치, 가상환경 설정
1. Miniconda 설치 (Python 배포 및 패키지 관리 도구)
   1. 공식 다운로드: Miniconda 설치 페이지 
      1. https://www.anaconda.com/docs/getting-started/miniconda/main
      2. OS에 맞는 버전 설치 (Windows/macOS)
   2. 설치 후 터미널/명령프롬프트에서 확인:
      1. conda --version
2. 가상환경 만들기 (프로젝트별 독립 환경)
   1. 가상환경 만들기
      1. (base) 환경에서 진행
      2. > conda create -n mktclass python=3.13
   2. 가상환경 목록
      1. > conda env list
   3. 가상환경 활성화/비활성화
      1. > conda activate mktclass
      2. > conda deactivate
   4. 가상환경 삭제
      1. > conda remove -n mktclass --all
   5. konlpy, mecab 사용할경우
      1. python 3.9버전 으로 가상환경 만들어야함
3. 가상환경에 패키지(라이브러리) 설치
   1. 패키지 종류
      1. 표준라이브러리: os, sys 등 파이썬 설치시 자동으로 설치
      2. 외부 패키지: numpy, pandas 등 필요에 따라 직접설치
   2. 외부 패키지 설치
      1. > conda activate mktclass
      2. > conda install 패키지
      3. > pip install 패키지
      4. conda install로 먼저 시도하고, 없으면 pip install 사용
   3. 설치된 외부 패키지 확인
      1. > conda list
      2. > pip list

# VS Code 설치, 설정
4. VS Code 설치 (코드 편집기)
   1. 공식 사이트: https://code.visualstudio.com
   2. 파이썬 entension 설치
      1. 좌측 메뉴에서 extensions --> Python 검색하여 설치
5. VS Code에서 Conda 가상환경 연동
   1. VS Code에서 Ctrl+Shift+P (Cmd+Shift+P on Mac)
   2. Python: Select Interpreter 입력
   3. 사용할 conda 가상환경 선택
6. extension 설치 (선택)
   1. Rainbow CSV, 
   2. vscode-icons, 
   3. vscode-pdf, 
   4. color themes 등 필요에 따라 설치

# yelp data 내려받기
7. Yelp Dataset
   1. https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/versions/6

# 코드 실행
8. 코드 라인별 실행 및 결과보기

방법1: 터미널에서 보기
- settings에서 "Send To Native Repl" 검색하여, 다음 항목 비활성화: "Send To Native REPL"
- settings에서 "Jupyter Send" 검색하여, 다음 항목 비활성화: "When pressing shift+enter, send code in a Python file to the Jupyter interactive window..."

방법2: Native REPL에서 보기
- settings에서 "Send To Native Repl" 검색하여, 다음 항목 활성화: "Send To Native REPL"

방법3: Interactive window에서 보기
- settings에서 "Send To Native Repl" 검색하여, 다음 항목 비활성화: "Send To Native REPL"
- settings에서 "Jupyter Send" 검색하여, 다음 항목 활성화: "When pressing shift+enter, send code in a Python file to the Jupyter interactive window..."
- ipykernel 설치 필요: pip install ipykernel

9. 파일 실행 (전체코드 한꺼번에 실행)
- 우측 상단 실행 버튼 (삼각형 모양 아이콘, Run Python File)


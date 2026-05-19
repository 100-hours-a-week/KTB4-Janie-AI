# 출결기록 저장, 조회, 출석 시 부트캠프에 필요한 사이트 빠르게 접속
# 정규 교육시간 이후 복습을 위한 뽀모도로 타이머 제공

import time
import datetime
import webbrowser   
import os

# 부트캠프 사이트 리스트
sites = [
    "https://kakao-tech-bootcamp.goorm.io/",
    "https://www.notion.com/ko",
    "https://www.github.com",
    "https://colab.research.google.com/",
    "https://discord.com/"
]

# 출결
record_file = 'attendance.txt'

def record_attendance(record_type):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with open(record_file, 'a', encoding='utf-8') as f:
        f.write(f"{date_str} {time_str} - {record_type}\n")
    
    # 출석 시 사이트 열기
    if record_type == "출석":
        for site in sites:
            webbrowser.open(site)

def view_records():
    if not os.path.exists(record_file):
        print("출결 기록 X")
        return
    
    with open(record_file, 'r', encoding='utf-8') as f:
        print(f.read())

# 타이머
def pomodoro_timer(minutes):
    try:
        for i in range(minutes * 60, 0, -1):
            mins, secs = divmod(i, 60)
            print(f"\r남은 시간: {mins:02d}:{secs:02d}", end="", flush=True)
            time.sleep(1)
        print('타이머 종료 !')
    except KeyboardInterrupt:
        print('\n타이머 중단 !')


while True:
    now = datetime.datetime.now()
    print(f"\n현재 시간: {now.strftime('%H:%M')}")
    
    if now.hour == 8 and now.minute >= 50:
        print("출석 체크 !")
    elif now.hour == 17 and now.minute >= 50:
        print("퇴실 체크 !")
    elif now.hour >= 18:
        print("<복습 시간> 타이머 실행 +_+ ")
    
    print("\n[on] 출석  [off] 퇴실  [rec] 출결기록 [tm] 타이머 [q] 종료")
    choice = input("입력: ").strip().lower()
    
    if choice == "q":
        print("프로그램 종료")
        break
    elif choice == "on":
        record_attendance("출석")
    elif choice == "off":
        record_attendance("퇴실")
    elif choice == "rec":
        view_records()
    elif choice == 'tm':
        try:
            minutes = int(input('시간(분) 설정:'))
            pomodoro_timer(minutes)
        except ValueError:
            print("입력 오류")
    else:
        print("입력 오류 ")


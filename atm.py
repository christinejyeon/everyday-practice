account = [1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000]

def atm():
    accNum = 0
    while accNum in [0,1,2,3,4,5,6,7,8,9]:
        accNum = int(input("계좌번호를 입력해주세요.: "))
        if accNum not in [0,1,2,3,4,5,6,7,8,9]:
            print("존재하지 않는 계좌번호입니다. 다시 입력해주세요.")
            accNum=0
        else:
            menuNum = 1
            while menuNum in [1,2,3,4,5]:
                menuNum = int(input("원하시는 작업을 선택해주세요.\n1:잔액확인 2:출금 3:입금 4:이체 5:이자율확인 6:종료\n"))
                if menuNum == 1:
                    print("잔액은 " + str(account[accNum])+ " 원 입니다.")
                elif menuNum == 2:
                    withdraw = int(input("출금하실 금액을 입력해주세요.: "))
                    if withdraw > account[accNum]:
                        print("잔액이 부족합니다. 메뉴 선택 화면으로 돌아갑니다.")
                    else:
                        account[accNum] = account[accNum] - withdraw
                        print("잔액은 " + str(account[accNum])+ " 원 입니다.")
                elif menuNum == 3:
                    deposit = int(input("입금하실 금액을 입력해주세요.: "))
                    account[accNum] = account[accNum] + deposit
                    print("잔액은 " + str(account[accNum])+ " 원 입니다.")
                elif menuNum == 4:
                    accTransfer = int(input("이체하실 계좌번호를 입력해주세요.: "))
                    if accTransfer == accNum:
                        print("자기 자신에게는 이체할 수 없습니다. 메뉴 선택 화면으로 돌아갑니다.")
                    elif accTransfer not in [0,1,2,3,4,5,6,7,8,9]:
                        print("존재하지 않는 계좌번호입니다. 메뉴 선택 화면으로 돌아갑니다.")
                    else:
                        transfer = int(input("이체하실 금액을 입력해주세요.: "))
                        if transfer > account[accNum]:
                            print("잔액이 부족합니다. 메뉴 선택 화면으로 돌아갑니다.")
                        else:
                            account[accNum] = account[accNum] - transfer
                            account[accTransfer] = account[accTransfer] + transfer
                            print("계좌번호 " + str(accTransfer) + " 로 " + str(transfer) + " 원 이체되었습니다.")
                            print("잔액은 " + str(account[accNum]) + " 원 입니다.")
                elif menuNum == 5:
                    print("이자율은 0.1% 입니다.")
                elif menuNum == 6:
                    print("종료를 선택하셨습니다. 안녕히 가세요.")

atm()

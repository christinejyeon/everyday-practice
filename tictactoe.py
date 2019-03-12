from tkinter import *

list = []

player="X"
turn=0
window = Tk()
window.title("틱택토")
status = Label(window, text="Status")

class cellState(object):
    X = "X"
    O = "O"
    empty = "0"

def checked(i):
    global player
    button = list[i]
    if button["text"] != " ":
        return
    button["text"] = player
    button["bg"] = "yellow"
    if player=="X":
        player = "O"
        button["bg"] = "yellow"
    else:
        player = "X"
        button["bg"] = "lightgreen"
    if (button0["text"]=="X" and button1["text"]=="X" and button2["text"]=="X") or (button0["text"]=="X" and button3["text"]=="X" and button6["text"]=="X") or (button0["text"]=="X" and button4["text"]=="X" and button8["text"]=="X") or (button1["text"]=="X" and button4["text"]=="X" and button7["text"]=="X") or (button2["text"]=="X" and button4["text"]=="X" and button6["text"]=="X") or (button2["text"]=="X" and button5["text"]=="X" and button8["text"]=="X") or (button3["text"]=="X" and button4["text"]=="X" and button5["text"]=="X") or (button6["text"]=="X" and button7["text"]=="X" and button8["text"]=="X"):
        status["text"] = "이겼다!"
    elif (button0["text"]=="O" and button1["text"]=="O" and button2["text"]=="O") or (button0["text"]=="O" and button3["text"]=="O" and button6["text"]=="O") or (button0["text"]=="O" and button4["text"]=="O" and button8["text"]=="O") or (button1["text"]=="O" and button4["text"]=="O" and button7["text"]=="O") or (button2["text"]=="O" and button4["text"]=="O" and button6["text"]=="O") or (button2["text"]=="O" and button5["text"]=="O" and button8["text"]=="O") or (button3["text"]=="O" and button4["text"]=="O" and button5["text"]=="O") or (button6["text"]=="O" and button7["text"]=="O" and button8["text"]=="O"):
        status["text"] = "ㅠㅠ져부렀어"
    elif button0["text"]!=" " and button1["text"]!=" " and button2["text"]!=" " and button3["text"]!=" " and button4["text"]!=" " and button5["text"]!=" " and button6["text"]!=" " and button7["text"]!=" " and button8["text"]!=" ":
        status["text"] = "비겼네요~"
    


player = "X"
for i in range(9):
            button = Button(window, text=" ", command=lambda k=i: checked(k))
            button.grid(row=i//3, column=i%3)
            list.append(button)           

status.grid(row=4,column=1)

#!/usr/bin/pyton3
import itertools
import numpy as np
import cv2

play1 = 1
play2 = 2
qTable = {}
episode = 0

for i in itertools.product([0, 1, 2], repeat=9):
    qTable.update({i: 0})


def reward(state):  # return reward in current state
    board = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for (a, b, c) in board:
        if state[a] != 0:
            if state[a] == state[b] == state[c] == 1:
                return 1
            if state[a] == state[b] == state[c] == 2:
                return -1
    return 0


def chooseMove(board, pool, player):  # return random position in space board
    temp = []
    for i in range(len(board)):
        if (board[i] == 0):
            temp.append(i)
    length = len(temp)
    return (temp[np.random.randint(0, length)])


def nextQvalue(board, pool, player):  # return all q-value in nextState for current state
    if (0 in board):
        value = []
        for i in range(len(board)):
            if (board[i] != 0):
                temp = board[:]
                temp[i] = player
                k = tuple(temp)
                value.append(pool[k])
        return value
    else:
        return [0, 0]


def feedBackQvalue(length, qTable, state):
    preState = state[:]
    while (length > 0):
        preState[p[length]] = 0
        qTable[tuple(preState)] = qTable[tuple(preState)] + 0.1 * (0.9 * qTable[tuple(state)] - qTable[tuple(preState)])
        state = preState[:]
        length -= 1
    return 0


while episode < 10000:
    episode += 1

    state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    p = []
    turn = 1
    while 0 in state:
        position = chooseMove(state, qTable, turn)  # return random position in space board
        p.append(position)  # recode path choose
        state[position] = turn
        if reward(state) == 1:
            if turn == 1:
                qTable[tuple(state)] = 1
            elif turn == 2:
                qTable[tuple(state)] = -1
            break
        elif turn == 1:
            turn = 2
            t = nextQvalue(state, qTable, turn)  # q-value
            expected = (0.9 * min(t) - qTable[tuple(state)])  # q-value
            qTable[tuple(state)] += 0.1 * expected  # q-value
        else:  # turn == 2
            turn = 1
            t = nextQvalue(state, qTable, turn)  # q-value
            expected = (0.9 * max(t) - qTable[tuple(state)])  # q-value
            qTable[tuple(state)] += 0.1 * expected  # q-value
    length = len(p) - 1
    k = feedBackQvalue(length, qTable, state)


def choose(board, pool):  # computer always to  choose maxQvalue position return
    k = -1000
    co = 0
    for i in range(len(board)):
        temp = board[:]
        if (board[i] == 0):
            temp[i] = 1
            if (pool[tuple(temp)] > k):
                k = pool[tuple(temp)]
                co = i
    return co


def cross(img, x, y):
    offset = 20
    distance = [(offset, offset),
                (offset, -1 * offset),
                (-1 * offset, -1 * offset),
                (-1 * offset, offset)]
    for ds in distance:
        cv2.line(img, (x, y), (x + ds[0], y + ds[1]), (255, 255, 255), 2)


def call_print(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        t = 0
        if x < 1 / 3 * weigh:
            if y < 1 / 3 * high:
                t = 0
            elif 1 / 3 * high < y < 2 / 3 * high:
                t = 3
            elif y > 2 / 3 * high:
                t = 6
        elif 1 / 3 * weigh < x < 2 / 3 * weigh:
            if y < 1 / 3 * high:
                t = 1
            elif 1 / 3 * high < y < 2 / 3 * high:
                t = 4
            elif y > 2 / 3 * high:
                t = 7
        elif x > 2 / 3 * weigh:
            if y < 1 / 3 * high:
                t = 2
            elif 1 / 3 * high < y < 2 / 3 * high:
                t = 5
            elif y > 2 / 3 * high:
                t = 8
        if param[0][t] == 2:
            pass
        else:
            cv2.circle(img, param[1][t], 50, (255, 255, 255), 3)
            param[0][t] = 2
            if 0 in board:
                postion = choose(param[0], qTable)
                board[postion] = 1
                cross(img, *locate[postion])
            if reward(board) == 1:
                cv2.putText(img, "Computer Win", (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            elif reward(board) == -1:
                cv2.putText(img, "You Win", (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            elif not 0 in board:
                cv2.putText(img, "Tie", (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("test", img)


img = cv2.imread("black.png", 0)
high, weigh = img.shape
lines = [[(int(weigh / 3), 0), (int(weigh / 3), high)],
         [(int(weigh * 2 / 3), 0), (int(weigh * 2 / 3), high)],
         [(0, int(high / 3)), (weigh, int(high / 3))],
         [(0, int(high * 2 / 3)), (weigh, int(high * 2 / 3))]]

for x, y in lines:
    cv2.line(img, x, y, (255, 255, 255), 2)

locate = []
offset = 60
t = 1
for j in range(1, 4):
    for i in range(1, 4):
        locate.append((int(i / 3 * weigh - offset), int(t / 3 * high - offset)))
    t = t + 1


board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# postion = choose(board, qTable)
# board[postion] = 1
# cross(img, *locate[postion])
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.setMouseCallback("test", call_print, [board, locate])
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




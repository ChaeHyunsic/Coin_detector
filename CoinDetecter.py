import numpy as np, cv2

def preprocessing(coin_no):
    fname = "coin/{0:02d}.png".format(coin_no)
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    if image is None: return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 명암도 영상 변환
    gray = cv2.GaussianBlur(gray, (7, 7), 2, 2)         # 블러링
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU          # 이진화
    _, th_img = cv2.threshold(gray, 130, 255, flag)

    mask = np.ones((3, 3), np.uint8)
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask)  # 열림 연산

    return image, th_img

def find_coins(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    circles = [cv2.minEnclosingCircle(c) for c in contours]  # 외각을 둘러싸는 원 검출
    circles = [(tuple(map(int, center)), int(radius)) for center, radius in circles if radius > 25]

    return circles

def make_coin_img(src, circles):
    coins = []
    for center, radius in circles:
        r = radius * 3
        cen = (r // 2, r // 2)                                      # 마스크 중심
        mask = np.zeros((r, r, 3), np.uint8)                        # 마스크 행렬
        cv2.circle(mask, cen, radius, (255, 255, 255), cv2.FILLED)

        coin = cv2.getRectSubPix(src, (r, r), center)
        coin = cv2.bitwise_and(coin, mask)                          # 마스킹 처리
        coins.append(coin)                                          # 동전 영상 저장

    return coins

def calc_histo_hue(coin):
    hsv = cv2.cvtColor(coin, cv2.COLOR_BGR2HSV)             # 컬러 공간 변환
    hsize, ranges = [32], [0, 180]                          # 32개 막대, 화소값 0~180 범위
    hist = cv2.calcHist([hsv], [0], None, hsize, ranges)

    return hist.flatten()

def grouping(hists):
    ws = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0]  # 가중치 지정

    sim = np.multiply(hists, ws)
    similaritys = np.sum(sim, axis=1) / np.sum(hists, axis=1)

    groups = [1 if s > 1.2 else 0 for s in similaritys]

    return groups

def classify_coins(circles, groups):
    ncoins = [0] * 4
    g = np.full((2,70), -1, int)

    g[0, 26:47], g[0, 47:50], g[0, 50:] = 0, 2, 3
    g[1, 36:44], g[1, 44:50], g[1, 50:] = 1, 2, 3

    for group, (_, radius) in zip(groups, circles):
        coin = g[group, radius]
        ncoins[coin] += 1

    return np.array(ncoins)

def put_string(frame, text, pt, value=None, color=(120, 200, 90)) :     # 프레임에 문자열 출력
    text = str(text) + str(value)
    shade = (pt[0] + 2, pt[1] + 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, text, shade, font, 0.7, (0, 0, 0), 2)    # 그림자 효과
    cv2.putText(frame, text, pt, font, 0.7, color, 2)           # 작성 문자

coin_no, max_no, cnt = 0, 85, 1

while True:
    coin_no = coin_no + cnt
    image, th_img = preprocessing(coin_no)

    if image is None:
        print("%02d.jpg: 영상 파일 없음" % coin_no)
        if coin_no < 0: coin_no = max_no
        elif coin_no >= max_no: coin_no = 0
        continue

    circles = find_coins(th_img)                                # 객체 검출
    coin_imgs = make_coin_img(image, circles)                   # 동전 영상 생성
    coin_hists= [calc_histo_hue(coin) for coin in coin_imgs]    # 동전 영상 히스토그램

    groups = grouping(coin_hists)                               # 동전 영상 그룹 분리
    ncoins = classify_coins(circles, groups)                    # 동전 인식
    coin_value = np.array([10, 50, 100, 500])                   # 동전 금액
    total = sum(coin_value * ncoins )                           # 화면에 인식되는 모든 동전의 금액합
    print_total = "Total : {:,} Won".format(total)
    put_string(image, print_total, (650, 50), '', (0,230,0))

    color = [(0, 0, 250), (255, 255, 0), (0, 250, 0), (250, 0, 255)]        # 동전별 색상
    for i, (c, r) in enumerate(circles):
        cv2.circle(image, c, r, color[groups[i]], 2)
        put_string(image, i + 1, (c[0] - 11, c[1] + 7), '', color[2])       # 동전에 검출 순번 표시

    cv2.imshow("CoinDetecter", image)

    key = cv2.waitKey(0)  # 키 이벤트 대기
    if key == 13 or key == 32: cnt = 1              # 엔터 키 이벤트나 스페이스 키 이벤트가 발생하면 다음 영상
    elif key == 8: cnt = -1                         # 백스페이스 키 이벤트가 발생하면 이전 영상
    elif key == 27: break                           # ESC 키 이벤트가 발생하면 종료

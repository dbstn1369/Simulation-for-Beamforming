import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 이미지 파일 경로
img_path1 = 'Time per Episode for Two Algorithms(100).png'
img_path2 = 'Total STS per Episode for Two Algorithms(100).png'

# 이미지 파일 불러오기
img1 = mpimg.imread(img_path1)
img2 = mpimg.imread(img_path2)

# 첫 번째 이미지 보여주기
plt.figure(figsize=(10, 5))  # 그림의 크기를 설정합니다.

plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 위치에 이미지를 보여줍니다.
plt.imshow(img1)
plt.axis('off')  # 축을 보여주지 않습니다.

# 두 번째 이미지 보여주기
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 위치에 이미지를 보여줍니다.
plt.imshow(img2)
plt.axis('off')  # 축을 보여주지 않습니다.

plt.tight_layout()  # 그림 간의 간격을 자동으로 조절합니다.
plt.show()

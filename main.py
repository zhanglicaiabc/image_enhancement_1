import numpy as np
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
from PIL import  Image
gray_level=256

def pixel_probability(img):
    assert isinstance(img, np.ndarray)
    prob = np.zeros(shape=(256))
    for rv in img:
        for cv in rv:
            prob[cv] += 1
    r, c = img.shape
    prob = prob / (r * c)
    return prob

def probability_to_histogram(img, prob):
    prob = np.cumsum(prob)  # 累计概率
    img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射
   # 像素值替换
    assert isinstance(img, np.ndarray)
    r, c = img.shape
    for ri in range(r):
        for ci in range(c):
            img[ri, ci] = img_map[img[ri, ci]]
    return img

def plot(y, name):
    plt.figure(num=name)
    plt.bar([i for i in range(gray_level)], y, width=1)

if __name__ == '__main__':

    #project1-1
    # img = np.array(imgplt.imread('0.jpg'))
    # prob = pixel_probability(img)
    # plot(prob, "原图直方图")
    # # 直方图均衡化
    # img = probability_to_histogram(img, prob)
    # #scipy.misc.imsave('source_hist.jpg', img)
    # im = Image.fromarray(img)
    # im.save("project1_1.jpeg")
    # prob = pixel_probability(img)
    # plot(prob, "直方图均衡化结果")
    # plt.show()

    #project2
    img=np.array(imgplt.imread('272.jpeg'))
    # filter=np.array([[-1,-1,-1],
    #                 [-1,8,-1],
    #                 [-1,-1,-1]])
    filter=np.zeros([5,5])
    result=np.zeros((img.shape[0]-filter.shape[0]+1,img.shape[1]-filter.shape[1]+1))
    print(img.shape)
    print(result.shape)
    for i in range(220):
        for j in range(220):
            # average=abs(img[i,j]*filter[0,0]+img[i,j+1]*filter[0,1]+img[i,j+2]*filter[0,2]
            #             +img[i+1,j]*filter[1,0]+img[i+1,j+1]*filter[1,1]+img[i+1,j+2]*filter[1,2]
            #             +img[i+2,j]*filter[2,0]+img[i+2,j+1]*filter[2,1]+img[i+2,j+2]*filter[2,2])
            # if(average)<200:
            #     result[i,j]=img[i+1,j+1]
            # else:
            #     result[i,j]=int(abs(img[i,j]*filter[0,0]+img[i,j+1]*filter[0,1]+img[i,j+2]*filter[0,2]
            #             +img[i+1,j]*filter[1,0]+img[i+1,j+2]*filter[1,2]
            #             +img[i+2,j]*filter[2,0]+img[i+2,j+1]*filter[2,1]+img[i+2,j+2]*filter[2,2])/5.2)
            for k in range(filter.shape[0]):
                for l in range(filter.shape[1]):
                    filter[k,l]=img[i-1+k][j-1+l]
            result[i][j]=np.median(filter)
    print(result)
    #result.astype(np.int16)
    im = Image.fromarray(result)
    im.convert('L').save("project1_4.jpeg")
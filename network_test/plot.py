import matplotlib.pyplot as plt
arr = [36.51685393258427,42.13483146067416, 42.69662921348314,47.752808988764045,48.31460674157304, 48.87640449438202, 49.43820224719101, 46.06741573033708,48.31460674157304]
x = [i for i in range(0, len(arr))]
plt.plot(x, arr)
plt.xlabel('round')
plt.ylabel('accuracy')
plt.savefig('res.png')
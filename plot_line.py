import pickle
import matplotlib.pylab as plt

def get_lists(path):
    with open(annotation_path,'rb') as f:
        annotation_dictionary = pickle.load(f)
    lists = sorted(annotation_dictionary.items()) # sorted by key, return a list of tuples
    return lists

prefix = 'f5_153'

#psnr lists (x,y)
annotation_path =  'psnr_annotation_backup/{}_psnr_match_index.pkl'.format(prefix)
psnr_list = get_lists(annotation_path)
x, y = zip(*psnr_list) # unpack a list of pairs into two tuples
print(psnr_list)
print("*******************************************************************************************")

#mse lists (a,b)
annotation_path =  'mse_annotation_backup/{}_mse_match_index.pkl'.format(prefix)
mse_list = get_lists(annotation_path)
a, b = zip(*mse_list) # unpack a list of pairs into two tuples
print(mse_list)
print("*******************************************************************************************")

#nrmse lists (c,d)
annotation_path =  'nrmse_annotation_backup/{}_nrmse_match_index.pkl'.format(prefix)
nrmse_list = get_lists(annotation_path)
c, d = zip(*nrmse_list) # unpack a list of pairs into two tuples
print(nrmse_list)
print("*******************************************************************************************")

# ssim lists (g,h)
annotation_path =  '{}_ssim_match_index.pkl'.format(prefix)
ssim_list = get_lists(annotation_path)
g, h = zip(*ssim_list) # unpack a list of pairs into two tuples
print(nrmse_list)
print("*******************************************************************************************")

# hfen lists (e,f)
annotation_path =  'hfen_annotation_backup/{}_hfen_match_index.pkl'.format(prefix)
hfen_list = get_lists(annotation_path)
e, f = zip(*hfen_list) # unpack a list of pairs into two tuples
# print(hfen_list)
print("*******************************************************************************************")

plt.plot(x, y, label = "psnr_index",alpha=0.7, linestyle = 'dashed')
plt.plot(a, b, label = "mse_index",alpha=0.7,linestyle = 'dashed')
plt.plot(c, d, label = "nrmse_index",alpha=0.7,linestyle = 'dashed')
plt.plot(e, f, label = "hfen_index",alpha=0.7,linestyle = 'dashed')
plt.plot(g, h, label = "ssim_index",alpha=0.7,linestyle = 'dashed')
plt.legend()
plt.show()
plt.savefig("line_plot.png")

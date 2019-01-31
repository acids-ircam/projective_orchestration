import numpy as np
import matplotlib.pyplot as plt

if __name__ =="__main__":
	path_to_model='/Users/leo/Recherche/lop/LOP/Experiments/NADE/1st_ordering/0/DEBUG/preds_nade/11'
	ordering = '1'
	batch_ind = 1
	# Load npy and concatenate along sample
	plot_mat = np.zeros((650,604))
	for ind_d in range(604):
		full_path = path_to_model + '/' + str(ind_d) + '_' + ordering + '.npy'
		aaa=np.load(full_path)
		plot_mat[ind_d]=aaa[batch_ind]
	# Add truth for the last layer
	full_path = path_to_model + '/truth.npy'
	aaa = np.load(full_path)
	plot_mat[-20:] = aaa[batch_ind]
	plt.imshow(plot_mat.T)
	plt.show()
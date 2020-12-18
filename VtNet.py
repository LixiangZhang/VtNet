if __name__ == '__main__': 
	import numpy as np
	import tensorflow as tf
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from sklearn.model_selection import cross_val_score
	from sklearn.model_selection import KFold
	from sklearn.feature_selection import mutual_info_regression
	from scipy.stats.stats import pearsonr
	import random
	from sklearn import svm
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn import tree
	from IPython.display import Image
	import pydotplus
	from numpy import genfromtxt
	import csv
	from sklearn import preprocessing
	import keras
	from keras.models import Sequential
	from keras.layers import Dense, Dropout
	from keras.optimizers import RMSprop
	from keras.utils import to_categorical
	from sklearn.metrics.classification import accuracy_score
	import pandas as pd
	from sklearn.metrics import roc_curve,auc,roc_auc_score
	import matplotlib.pyplot as plt
	from scipy import interp
	import pandas as pd
	from sklearn.preprocessing import label_binarize
	from sklearn.multiclass import OneVsRestClassifier
	import cdt
	from cdt import SETTINGS
	SETTINGS.verbose=False
	SETTINGS.NJOBS=16
	import networkx as nx
	import time
	from cdt.causality.graph import CGNN
	from cdt.independence.graph import FSGNN
	from cdt.causality.pairwise import GNN
	from cdt.utils.graph import dagify_min_edge
	import pandas as pd
	import sys

	#Functions defined for generating the minimum spanning tree on directed graphs
	# --------------------------------------------------------------------------------- #

	def _reverse(graph):
		r = {}
		for src in graph:
			for (dst,c) in graph[src].items():
				if dst in r:
					r[dst][src] = c
				else:
					r[dst] = { src : c }
		return r

	# Finds all cycles in graph using Tarjan's algorithm
	def strongly_connected_components(graph):
		"""
		Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
		for finding the strongly connected components of a graph.
		"""

		index_counter = [0]
		stack = []
		lowlinks = {}
		index = {}
		result = []

		def strongconnect(node):
			# set the depth index for this node to the smallest unused index
			index[node] = index_counter[0]
			lowlinks[node] = index_counter[0]
			index_counter[0] += 1
			stack.append(node)

			# Consider successors of `node`
			try:
				successors = graph[node]
			except:
				successors = []
			for successor in successors:
				if successor not in lowlinks:
					# Successor has not yet been visited; recurse on it
					strongconnect(successor)
					lowlinks[node] = min(lowlinks[node],lowlinks[successor])
				elif successor in stack:
					# the successor is in the stack and hence in the current strongly connected component (SCC)
					lowlinks[node] = min(lowlinks[node],index[successor])

			# If `node` is a root node, pop the stack and generate an SCC
			if lowlinks[node] == index[node]:
				connected_component = []

				while True:
					successor = stack.pop()
					connected_component.append(successor)
					if successor == node: break
				component = tuple(connected_component)
				# storing the result
				result.append(component)

		for node in graph:
			if node not in lowlinks:
				strongconnect(node)

		return result

	def _mergeCycles(cycle,G,RG,g,rg):
		allInEdges = [] # all edges entering cycle from outside cycle
		minInternal = None
		minInternalWeight = sys.maxsize

		# Find minimal internal edge weight
		for n in cycle:
			for e in RG[n]:
				if e in cycle:
					if minInternal is None or RG[n][e] < minInternalWeight:
						minInternal = (n,e)
						minInternalWeight = RG[n][e]
						continue
				else:
					allInEdges.append((n,e)) # edge enters cycle

		# Find the incoming edge with minimum modified cost
		# modified cost c(i,k) = c(i,j) - (c(x_j, j) - min{j}(c(x_j, j)))
		minExternal = None
		minModifiedWeight = 0
		for j,i in allInEdges: # j is vertex in cycle, i is candidate vertex outside cycle
			xj, weight_xj_j = rg[j].popitem() # xj is vertex in cycle that currently goes to j
			rg[j][xj] = weight_xj_j # put item back in dictionary
			w = RG[j][i] - (weight_xj_j - minInternalWeight) # c(i,k) = c(i,j) - (c(x_j, j) - min{j}(c(x_j, j)))
			if minExternal is None or w <= minModifiedWeight:
				minExternal = (j,i)
				minModifiedWeight = w

		w = RG[minExternal[0]][minExternal[1]] # weight of edge entering cycle
		xj,_ = rg[minExternal[0]].popitem() # xj is vertex in cycle that currently goes to j
		rem = (minExternal[0], xj) # edge to remove
		rg[minExternal[0]].clear() # popitem() should delete the one edge into j, but we ensure that

		# Remove offending edge from RG
		# RG[minExternal[0]].pop(xj, None) #highly experimental. throw away the offending edge, so we never get it again

		if rem[1] in g:
			if rem[0] in g[rem[1]]:
				del g[rem[1]][rem[0]]
		if minExternal[1] in g:
			g[minExternal[1]][minExternal[0]] = w
		else:
			g[minExternal[1]] = { minExternal[0] : w }

		rg = _reverse(g)

	def mst(root,G):
		""" The Chu-Liu/Edmond's algorithm

		arguments:

		root - the root of the MST
		G - the graph in which the MST lies

		returns: a graph representation of the MST

		Explanation is copied verbatim here:

		The input graph G is assumed to have the following
		representation: A vertex can be any object that can
		be used as an index into a dictionary.  G is a
		dictionary, indexed by vertices.  For any vertex v,
		G[v] is itself a dictionary, indexed by the neighbors
		of v.  For any edge v->w, G[v][w] is the length of
		the edge.
		"""

		RG = _reverse(G)

		g = {}
		for n in RG:
			if len(RG[n]) == 0:
				continue
			minimum = sys.maxsize
			s,d = None,None

			for e in RG[n]:
				if RG[n][e] < minimum:
					minimum = RG[n][e]
					s,d = n,e

			if d in g:
				g[d][s] = RG[s][d]
			else:
				g[d] = { s : RG[s][d] }

		cycles = [list(c) for c in strongly_connected_components(g)]

		cycles_exist = True
		while cycles_exist:

			cycles_exist = False
			cycles = [list(c) for c in strongly_connected_components(g)]
			rg = _reverse(g)

			for cycle in cycles:

				if root in cycle:
					continue

				if len(cycle) == 1:
					continue

				_mergeCycles(cycle, G, RG, g, rg)
				cycles_exist = True

		return g

	#Read in data, using Yan data as an example
	# --------------------------------------------------------------------------------- #
	random.seed(1230)
	my_data = genfromtxt('block_yan.csv', delimiter=',')
	yan_label=my_data[:,0]-np.ones(len(my_data[:,0]))
	yan_label.astype(int)
	yan_data=my_data[:,1:484]

	X_train,X_test,y_train,y_test = train_test_split(yan_data,yan_label,test_size=0.3,random_state=1230)
	X_train=preprocessing.normalize(X_train,axis=0)
	X_test=preprocessing.normalize(X_test,axis=0)
	mlp_train=to_categorical(y_train)
	mlp_test=to_categorical(y_test)

	#Generating variable blocks according to the correlation
	# --------------------------------------------------------------------------------- #
	num_total_blocks=15
	num_feature_perblock=int(X_train.shape[1]/num_total_blocks)
	remain_feature=np.arange(0,X_train.shape[1],1)
	block_structure=[]
	for i in range(0,num_total_blocks-1):
		screen=np.zeros(remain_feature.shape[0])
		can=random.randint(0,remain_feature.shape[0]-1)
		for j in range(0,remain_feature.shape[0]):
			#screen[j]=-abs(mutual_info_regression(X_train[:,remain_feature[0]].reshape((-1, 1)),X_train[:,remain_feature[j]]))
			screen[j]=-abs(pearsonr(X_train[:,remain_feature[can]],X_train[:,remain_feature[j]])[0])
		sort_index = np.argsort(screen)[0:num_feature_perblock]
		in_block=remain_feature[sort_index]
		block_structure.append(in_block)
		remain_feature=np.setdiff1d(remain_feature,in_block)
	block_structure.append(remain_feature)

	#Generating the graph structure
	#This process might take a while
	# --------------------------------------------------------------------------------- #

	x_graph=np.zeros([X_train.shape[0],num_total_blocks])
	for i in range(num_total_blocks):
		x_graph[:,i]=np.mean(X_train[:,block_structure[i]],axis = 1)

	
	Fsgnn = FSGNN(train_epochs=20, test_epochs=10, l1=0.1, batch_size=20)
	data=pd.DataFrame(x_graph)
	ugraph = Fsgnn.predict(data, threshold=1e-7)
	#nx.draw_networkx(ugraph, font_size=8) # The plot function allows for quick visualization of the graph.
	#plt.show()
	gnn = GNN(nruns=12, train_epochs=20, test_epochs=10, batch_size=20)
	ograph = dagify_min_edge(gnn.orient_graph(data, ugraph))
	#nx.draw_networkx(ugraph, font_size=8) # The plot function allows for quick visualization of the graph.
	#plt.show()
	# List results
	#pd.DataFrame(list(ograph.edges(data='weight')), columns=['Cause', 'Effect', 'Score'])
	Cgnn = CGNN(nruns=12, train_epochs=20, test_epochs=10, batch_size=20)
	dgraph = Cgnn.orient_directed_graph(data, ograph)

	# Plot the output graph
	nx.draw_networkx(dgraph, font_size=8) # The plot function allows for quick visualization of the graph.
	plt.show() 
	# Print output results : 
	#pd.DataFrame(list(dgraph.edges(data='weight')), columns=['Cause', 'Effect', 'Score'])


	ed=np.array(list(dgraph.edges(data='weight')))
	G={}
	for i in range(ed.shape[0]):
		G[int(ed[i][0])]={}
	for i in range(ed.shape[0]):
		G[int(ed[i][0])][int(ed[i][1])]=-ed[i][2]

	G = mst(0,G)
	
	#G is the final tree structure for traing VtNet
	# --------------------------------------------------------------------------------- #
	print(G)

	#Mini-Batch
	# --------------------------------------------------------------------------------- #

	def gen_batch(raw_data_x,raw_data_y, batch_size, shuffle=True):
		if shuffle:
			indices = np.arange(raw_data_x.shape[0])
			np.random.shuffle(indices)
		for start_idx in range(0, raw_data_x.shape[0] - batch_size + 1, batch_size):
			if shuffle:
				excerpt = indices[start_idx:start_idx + batch_size]
			else:
				excerpt = slice(start_idx, start_idx + batch_size)
			yield (raw_data_x[excerpt,:], raw_data_y[excerpt])

	#LSTM-like VtNet cells
	# --------------------------------------------------------------------------------- #
	
	def lstm_cell(lstm_input,feature,state_size,state,cell):
		Wf = tf.Variable((tf.random_normal([feature + state_size, 2*state_size], stddev=0.1)))
		bf = tf.Variable(tf.zeros([2*state_size]),name="biasf")
		Wi = tf.Variable((tf.random_normal([feature + state_size, 2*state_size], stddev=0.1)))
		bi = tf.Variable(tf.zeros([2*state_size]),name="biasi")
		Wa = tf.Variable((tf.random_normal([feature + state_size, 2*state_size], stddev=0.1)))
		ba = tf.Variable(tf.zeros([2*state_size]),name="biasa")
		Wo = tf.Variable((tf.random_normal([feature + state_size, 2*state_size], stddev=0.1)))
		bo = tf.Variable(tf.zeros([2*state_size]),name="biaso")
		Wff = tf.Variable((tf.random_normal([2*state_size, state_size], stddev=0.1)))
		bff = tf.Variable(tf.zeros([state_size]),name="biasff")
		Wii = tf.Variable((tf.random_normal([2*state_size, state_size], stddev=0.1)))
		bii = tf.Variable(tf.zeros([state_size]),name="biasii")
		Waa = tf.Variable((tf.random_normal([2*state_size, state_size], stddev=0.1)))
		baa = tf.Variable(tf.zeros([state_size]),name="biasaa")
		Woo = tf.Variable((tf.random_normal([2*state_size, state_size], stddev=0.1)))
		boo = tf.Variable(tf.zeros([state_size]),name="biasoo")
		ff=tf.matmul(tf.matmul(tf.concat([lstm_input, state], 1), Wf) + bf,Wff)+bff
		f=tf.sigmoid(ff)
		ii=tf.matmul(tf.matmul(tf.concat([lstm_input, state], 1), Wi) + bi,Wii)+bii
		i=tf.sigmoid(ii)
		aa=tf.matmul(tf.matmul(tf.concat([lstm_input, state], 1), Wa) + ba,Waa)+baa
		a=tf.tanh(aa)
		oo=tf.matmul(tf.matmul(tf.concat([lstm_input, state], 1), Wo) + bo,Woo)+boo
		#oo = tf.nn.dropout(oo, keep_prob=0.8)  # DROP-OUT can be applied here is necessary
		o=tf.sigmoid(oo)
		c=tf.multiply(cell, f)+tf.multiply(i,a)
		h=tf.multiply(o,tf.tanh(c))
		return h,c,i,f

	#Function for using the tree struvture
	# --------------------------------------------------------------------------------- #

	def dfs(state,cell,G,node,lstm_input,block_structure,state_size):
		state,cell,i,f = lstm_cell(tf.transpose(tf.nn.embedding_lookup(tf.transpose(lstm_input), tf.constant(block_structure[node]))),len(block_structure[node]),state_size, state, cell)
		global input_gate
		global forget_gate
		input_gate[node]=i
		forget_gate[node]=f
		if node not in G:
			lstm_outputs.append(state)
			return
		else:
			if G[node]=={}:
				lstm_outputs.append(state)
				return
			for (dst,c) in G[node].items():
				dfs(state,cell,G,dst,lstm_input,block_structure,state_size)

	#Training the VtNet
	# --------------------------------------------------------------------------------- #

	batch_size = 20
	num_classes = 7
	num_epoch= 300
	state_size =20
	learning_rate = 0.001
	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [None, X_train.shape[1]], name='input_placeholder')
	y = tf.placeholder(tf.int32, [None], name='labels_placeholder')
	init_state = tf.placeholder(tf.float32,[None, state_size])
	init_cell = tf.placeholder(tf.float32,[None, state_size])
	lstm_input = x
	state = init_state
	cell = init_cell
	lstm_outputs = []
	cell_outputs = []
	input_gate = {}
	forget_gate = {}
	tail=set([dst for i in G for (dst,c) in G[i].items()])
	head=set([i for i in G])
	root=list(head-tail)
	for r in root:
		dfs(state,cell,G,r,lstm_input,block_structure,state_size)
	h_all=lstm_outputs[0]
	for l in range(1,len(lstm_outputs)):
		h_all=tf.concat([h_all,lstm_outputs[l]],axis=1)
	#The layer connecting softmax layer
	W = tf.Variable((tf.random_normal([state_size*len(lstm_outputs), num_classes], stddev=0.01)))
	b = tf.Variable(tf.zeros([num_classes]),name="biass")
	#h_all = tf.nn.dropout(h_all, keep_prob=0.8)  # DROP-OUT can be applied if necessary
	logit = tf.matmul(h_all, W) + b
	pred=tf.nn.sigmoid(logit)
	predictions = tf.argmax(pred,1,output_type=tf.int32)
	correct_prediction = tf.equal(predictions, y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#Penalty can be applied if necessary
	#vars   = tf.trainable_variables()
	losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit)#+tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * 0.01
	total_loss = tf.reduce_mean(losses)
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		training_losses = []
		training_state = np.zeros((batch_size, state_size))
		training_cell = np.zeros((batch_size, state_size))
		for i in range(0,num_epoch):
			print("\nEPOCH", i)
			training_loss = 0
			for step, (X,Y) in enumerate(gen_batch(X_train, y_train, batch_size)):
				tr_losses, training_loss_, trainging_accuracy_,  _ = \
						sess.run([losses,
								total_loss,
								accuracy,
								train_step],
									feed_dict={x:X, y:Y, init_state:training_state,init_cell:training_cell})
				training_loss =training_loss_
				trainging_accuracy=trainging_accuracy_
				training_losses.append(training_loss)
				if step % 2 == 0 and step > 0:
					print("Loss at step", step,
							"is:", training_loss, "and accuracy is:", trainging_accuracy)
				training_loss = 0
		test_state = np.zeros((y_test.shape[0], state_size))
		test_cell = np.zeros((y_test.shape[0], state_size))
		train_state = np.zeros((y_train.shape[0], state_size))
		train_cell = np.zeros((y_train.shape[0], state_size))
		in_gate, for_gate=sess.run([input_gate,forget_gate],feed_dict={x:X_train, y:y_train,init_state:train_state,init_cell:train_cell})
		acc=sess.run(accuracy,feed_dict={x:X_test, y:y_test,init_state:test_state,init_cell:test_cell})
		print("VtNet Accuracy of test data is:",acc)

	#Significance score
	# --------------------------------------------------------------------------------- #
	score={}
	for i in in_gate:
		score[i]=np.linalg.norm(in_gate[i])/np.linalg.norm(for_gate[i])

	print("Significant score for each node of training data is:",score)


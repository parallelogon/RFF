import torch
import numpy as np
from sklearn.datasets import make_circles
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl

def changeOfVariables(a,b):
	A = torch.from_numpy(a)
	B = torch.from_numpy(b)

	la,wa = A.shape
	lb,wb = B.shape
	Atransformed = np.repeat(A, lb, axis = 0).reshape(la,-1)
	out = (Atransformed + B.detach().numpy().ravel()).reshape(-1,wa)/2
	return(torch.tensor(out))


def HKGaussian(X,theta,omega):
	x = X.detach().numpy()
	Xavg = changeOfVariables(x,x)
	K1 = K(X,theta).view(-1,1)
	K2 = K(X,theta).view(-1,1)
	K3 = K(Xavg,omega)

	return((K1 @ K2.T)*K3)

def Z(X, Theta):
	return(torch.tensor(2.0/len(Theta)).sqrt()*torch.cos(Theta @ X.T))

def K(X, Theta):
	z = Z(X, Theta)
	return(z.T @ z)


def create_mini_batches(X, y, batch_size):
	mini_batches = []
	data = np.hstack((X, y))
	np.random.shuffle(data)
	n_minibatches = data.shape[0] // batch_size
	i = 0

	for i in range(n_minibatches):# + 1):
		mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
		X_mini = mini_batch[:, :-1]
		Y_mini = mini_batch[:, -1].reshape((-1, 1))
		mini_batches.append((X_mini, Y_mini))

	if data.shape[0] % batch_size != 0:
		mini_batch = data[i * batch_size:data.shape[0]]
		X_mini = mini_batch[:, :-1]
		Y_mini = mini_batch[:, -1].reshape((-1, 1))
		mini_batches.append((X_mini, Y_mini))
	return mini_batches

def hingeLoss(kernelFunc, Y, alpha):
	yHat = kernelFunc @ (alpha)
	return torch.maximum(torch.zeros(len(Y), 1), 1 - Y*yHat)

def lossFn(xs, ys, X, theta, alpha, lbda):
	k = Z(xs, theta).T @ Z(X, theta)
	return(torch.mean(hingeLoss(k, ys, alpha)) + (lbda/2) * (alpha).T @ K(X, theta) @ (alpha))


class Clf:
	opt = False

	def __init__(self, X, Y, lbda = 1, r = 2**6):
		n,d = X.shape
		self.X = torch.tensor(X, dtype = torch.float)
		self.Y = torch.tensor(Y, dtype = torch.float)
		self.lbda = lbda
		self.theta = torch.randn((r, d), requires_grad=True)
		self.alpha = torch.randn((n, 1), requires_grad=True)
		self.alpha.data = self.alpha.data / (np.linalg.norm(self.alpha.data.detach().numpy())*2*np.sqrt(lbda))
		self.LL = []


	def SGDAlpha(self,loss, X, Y, Theta, alpha, lbda, num_epochs, block):
		for epoch in range(num_epochs):
			batches = create_mini_batches(X.detach().numpy(), Y.detach().numpy(), block)
			lAvg = 0
			for mini_batch in batches:
				xs, ys = mini_batch
				xs = torch.tensor(xs, dtype=torch.float)
				ys = torch.tensor(ys, dtype=torch.float)
				L = loss(xs,ys, X, Theta,alpha,lbda)
				ll = L.detach().numpy()[0][0]
				if np.isnan(ll) == False:
					lAvg += ll
				if L != 0:
					L.backward()
					self.alpha.data -= (1/(lbda*(epoch+1)))*(1/block)*alpha.grad.data
					self.alpha.grad.data.zero_()

				self.alpha.data = torch.minimum(torch.ones(
					len(self.alpha), 1), 1/(lbda*torch.norm(self.alpha.data, p=2)))*self.alpha.data
			self.LL.append(lAvg/len(batches))

	def SGDTheta(self,loss, X, Y, Theta, alpha, lbda, num_epochs, block):
		for epoch in range(num_epochs):
			batches = create_mini_batches(X.detach().numpy(), Y.detach().numpy(), block)
			#print("theta epoch: ", epoch+1)
			lAvg = 0
			for mini_batch in batches:
				xs, ys = mini_batch
				xs = torch.tensor(xs, dtype=torch.float)
				ys = torch.tensor(ys, dtype=torch.float)
				L = loss(xs,ys,X, Theta,alpha,lbda)
				ll = L.detach().numpy()[0][0]
				if np.isnan(ll) == False:
					lAvg += ll
				if L != 0:
					L.backward()
					self.theta.data -= (1/(lbda*(epoch+1)))*(1/block)*self.theta.grad.data
					self.theta.grad.data.zero_()
			self.LL.append(lAvg/len(batches))

	def Optimize(self, passes = 40, T =100, block = 10, loss = lossFn):
		self.opt = True
		for i in range(passes):
			print("Pass: ", i+1)
			self.SGDAlpha(loss, self.X, self.Y, self.theta, self.alpha, self.lbda, T, block)
			self.SGDTheta(loss, self.X, self.Y, self.theta, self.alpha, self.lbda, T, block)

			#results = self.Y*(K(self.X, self.theta)
			#            @ (self.alpha)).detach().numpy() > 0
			#accuracy = np.mean(results.detach().numpy())
			#if accuracy > 0.98:
			#    break

	def Predict(self, xNew):
		xx = torch.tensor(xNew, dtype = torch.float)
		return((Z(xx,self.theta).T @ Z(self.X, self.theta) @ self.alpha).detach().numpy())


	def Fit(self):
		p3 = plt.figure()
		plt.plot(self.LL)
		plt.show()

	def Show(self):
		if (self.opt == False) and (list(self.X.size())[1] <=3):
			p1 = plt.figure()
			plt.scatter(self.X.detach().numpy()[:, 1], self.X.detach().numpy()[:, 2], c=self.Y.detach().numpy().ravel() > 0)
			with mpl.rc_context(rc={'interactive': False}):
				plt.show()

		if (self.opt == True) and (list(self.X.size())[1] <=3):
			resolution = .05
			p2 = plt.figure()
			# setup marker generator and color map
			markers = ('s', 'x', 'o', '^', 'v')
			colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
			cmap = ListedColormap(colors[:len(np.unique(self.Y.detach().numpy()))])

			# plot the decision surface
			x1_min, x1_max = self.X.detach().numpy()[:, 1].min() - 1, self.X.detach().numpy()[:, 1].max() + 1
			x2_min, x2_max = self.X.detach().numpy()[:, 2].min() - 1, self.X.detach().numpy()[:, 2].max() + 1
			xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
								np.arange(x2_min, x2_max, resolution))
			grid = np.c_[xx1.ravel(), xx2.ravel()]
			grid = np.hstack((np.ones((len(grid), 1)), grid))
			Z = np.array([self.Predict(g.reshape(1,-1)) > 0 for g in grid])
			Z = Z.reshape(xx1.shape)
			plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
			plt.xlim(xx1.min(), xx1.max())
			plt.ylim(xx2.min(), xx2.max())

			for idx, cl in enumerate(np.unique(self.Y.detach().numpy())):
				plt.scatter(x=self.X.detach().numpy()[self.Y.detach().numpy().ravel() == cl, 1], y=self.X.detach().numpy()[self.Y.detach().numpy().ravel() == cl, 2],
							alpha=0.8, c=np.array(cmap(idx)),
							marker=markers[idx], label=cl)
			plt.legend()
			plt.show()

class hClf(Clf):
	def __init__(self, X, Y, lbda = 1, lbdaQ = 1, r = 2**6, l = 2**6):
		super().__init__(X, Y, lbda, r)
		n,d = X.shape
		self.beta = torch.randn((n**2,1), requires_grad = True)
		self.omega = torch.randn((l, d), requires_grad=True)
		self.lbdaQ = lbdaQ

	def Predict(self, xNew):
		#K = HKGaussian(self.X, self.theta, self.omega)

		if torch.is_tensor(xNew) == False:
			xx = torch.from_numpy(xNew)
		else:
			xx = xNew

		XnewAvg = changeOfVariables(xx.detach().numpy(),self.X.detach().numpy())
		XAvg = changeOfVariables(self.X.detach().numpy(),self.X.detach().numpy())

		s1 = (Z(xx.type(torch.float),self.theta).T @ Z(self.X, self.theta)).view(-1,1)
		s2 = (Z(self.X, self.theta).T @ Z(self.X, self.theta)).view(1,-1)
		s3 = Z(XnewAvg.type(torch.float), self.omega).T @ Z(XAvg.type(torch.float), self.omega)

		HKpred = (s1 @ s2)*s3
		kpred = (HKpred @ self.beta).view(-1, len(self.alpha))
		yHat = kpred @ self.alpha
		return(yHat)

	def hyperlossFn(self,xs, ys, X, theta, omega, alpha, beta, lbda, lbdaQ):
		
		K = HKGaussian(X,theta,omega)
		k = (K @ beta).view(len(alpha), -1)
		yHat = self.Predict(xs)

		loss = torch.maximum(torch.zeros(len(ys), 1), 1 - ys*yHat)
		return(torch.mean(loss) + (lbda/2)*alpha.T @ k @ alpha + (lbdaQ/2)*beta.T @ K @ beta)# + torch.norm(omega, p = 1) + torch.norm(theta, p = 1))

	def SGDAlphah(self,loss, X, Y, Theta, alpha, lbda, omega, beta, lbdaQ, num_epochs, block):
		for epoch in range(num_epochs):
			batches = create_mini_batches(X.detach().numpy(), Y.detach().numpy(), block)
			lAvg = 0
			for mini_batch in batches:
				xs, ys = mini_batch
				xs = torch.tensor(xs, dtype=torch.float)
				ys = torch.tensor(ys, dtype=torch.float)
				L = loss(xs,ys, X, Theta,omega, alpha,beta, lbda, lbdaQ)
				ll = L.detach().numpy()[0][0]
				if np.isnan(ll) == False:
					lAvg += ll
					if L != 0:
						L.backward()
						self.alpha.data -= 0.0001*(1/(lbda*(epoch+1)))*(1/block)*alpha.grad.data
						self.alpha.grad.data.zero_()
						self.alpha.data = torch.minimum(torch.ones(len(self.alpha), 1), 1/(lbda*torch.norm(self.alpha.data, p=2)))*self.alpha.data
			self.LL.append(lAvg/len(batches))

	def SGDBetah(self,loss, X, Y, Theta, alpha, lbda, omega, beta, lbdaQ, num_epochs, block):
		for epoch in range(num_epochs):
			batches = create_mini_batches(X.detach().numpy(), Y.detach().numpy(), block)
			lAvg = 0
			for mini_batch in batches:
				xs, ys = mini_batch
				xs = torch.tensor(xs, dtype=torch.float)
				ys = torch.tensor(ys, dtype=torch.float)
				L = loss(xs,ys, X, Theta,omega, alpha,beta, lbda, lbdaQ)
				ll = L.detach().numpy()[0][0]
				if np.isnan(ll) == False:
					lAvg += ll
					if L != 0:
						L.backward()
						self.beta.data -= 0.0001*(1/(lbdaQ*(epoch+1)))*(1/block)*beta.grad.data
						self.beta.grad.data.zero_()
						#self.beta.data = torch.minimum(torch.ones(
						#	len(self.beta), 1), 1/(lbdaQ*torch.norm(self.beta.data, p=2)))*self.beta.data
						self.beta.data = torch.maximum(torch.zeros(len(self.beta),1), self.beta.data)
			self.LL.append(lAvg/len(batches))

	def SGDThetah(self,loss, X, Y, Theta, alpha, lbda, omega, beta, lbdaQ, num_epochs, block):
		for epoch in range(num_epochs):
			batches = create_mini_batches(X.detach().numpy(), Y.detach().numpy(), block)
			#print("theta epoch: ", epoch+1)
			lAvg = 0
			for mini_batch in batches:
				xs, ys = mini_batch
				xs = torch.tensor(xs, dtype=torch.float)
				ys = torch.tensor(ys, dtype=torch.float)
				L = loss(xs,ys, X, Theta,omega, alpha,beta, lbda, lbdaQ)
				ll = L.detach().numpy()[0][0]
				if np.isnan(ll) == False:
					lAvg += ll
					if L != 0:
						L.backward()
						self.theta.data -= 0.0001*(1/(lbda*(epoch+1)))*(1/block)*self.theta.grad.data
						self.theta.grad.data.zero_()
						#self.theta.data = self.theta.data % (2*np.pi)
			self.LL.append(lAvg/len(batches))

	def SGDOmegah(self,loss, X, Y, Theta, alpha, lbda, omega, beta, lbdaQ, num_epochs, block):
		for epoch in range(num_epochs):
			batches = create_mini_batches(X.detach().numpy(), Y.detach().numpy(), block)
			#print("theta epoch: ", epoch+1)
			lAvg = 0
			for mini_batch in batches:
				xs, ys = mini_batch
				xs = torch.tensor(xs, dtype=torch.float)
				ys = torch.tensor(ys, dtype=torch.float)
				L = loss(xs,ys, X, Theta,omega, alpha,beta, lbda, lbdaQ)
				ll = L.detach().numpy()[0][0]
				if np.isnan(ll) == False:
					lAvg += ll
					if L != 0:
						L.backward()
						self.omega.data -= 0.0001*(1/(lbdaQ*(epoch+1)))*(1/block)*self.omega.grad.data
						self.omega.grad.data.zero_()
						#self.omega.data = self.omega.data % (2*np.pi)
			self.LL.append(lAvg/len(batches))

	def Optimize(self, passes = 40, T =100, block = 10, loss = hyperlossFn, es = False):
		self.opt = True

		if type(T) == int:
			T1 = T2 = T3 = T4 = T
		elif len(T) == 4:
			T1 = T[0]
			T2 = T[1]
			T3 = T[2]
			T4 = T[3]
			print(T1,T2,T3,T4)
		else:
			print("Bad T")

		for i in range(passes):
			print("Pass: ", i+1)
			print("Alpha")
			self.SGDAlphah(self.hyperlossFn, self.X, self.Y, self.theta, self.alpha, self.lbda, self.omega, self.beta, self.lbdaQ, T1, block)
			print("Beta")
			self.SGDBetah(self.hyperlossFn, self.X, self.Y, self.theta, self.alpha, self.lbda, self.omega, self.beta, self.lbdaQ, T2, block)
			print("Theta")
			self.SGDThetah(self.hyperlossFn, self.X, self.Y, self.theta, self.alpha, self.lbda, self.omega, self.beta, self.lbdaQ, T3, block)
			print("Omega")
			self.SGDOmegah(self.hyperlossFn, self.X, self.Y, self.theta, self.alpha, self.lbda, self.omega, self.beta, self.lbdaQ, T4, block)

			accuracy = np.mean( (self.Y*torch.sign(self.Predict(self.X))).detach().numpy() > 0)
			print("Accuracy: ", accuracy)
			if accuracy >= 0.98 and es == True:
				break


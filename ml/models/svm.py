import ml.models.base as ml
import numpy as np
import cvxpy as cvx

from tensorflow import keras
# from sklearn.svm import SVC, NuSVC
from thundersvm import *

class SVM(ml.Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.kernel = kwargs.get('kernel', "linear")
        self.C = kwargs.get('C', 0.5)
        self.nu = kwargs.get('nu', None)
        self.degree = kwargs.get('degree', 3)
        self.gamma = kwargs.get('gamma', 'auto')
        self.tol = kwargs.get('tol', 1e-3)
        self.coef0 = kwargs.get('coef0', 0)
        self.model = None
        self.verbose = 0

    def fit(self, X, Y, **kwargs):
        if self.nu is None:
            self.model = SVC(C=self.C,
                    kernel=self.kernel,
                    degree=self.degree,
                    gamma=self.gamma,
                    tol=self.tol,
                    coef0=self.coef0,
                    random_state=42,
                    # class_weight=self.get_class_weight(Y),
                    verbose=self.verbose)
        else:
            self.model = NuSVC(nu=self.nu,
                    kernel=self.kernel,
                    degree=self.degree,
                    gamma=self.gamma,
                    tol=self.tol,
                    coef0=self.coef0,
                    random_state=42,
                    # class_weight=self.get_class_weight(Y),
                    verbose=self.verbose)
        print(Y)
        self.model.fit(X,Y)

    def predict(self, X, **kwargs):
        if self.model is None:
            raise UnboundLocalError("it is not possible to predict the data without training")

        return self.model.predict(X)

class KC(ml.Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.kernel = kwargs.get('kernel', "polynomial")
        self.kernel_arg = kwargs.get('kernel_arg', 1)
        self.loss = kwargs.get('loss', 'logistic')
        self.params = None
        self.verbose = 0

    def transform(self, X):
        if self.kernel == 'polynomial':
            #k(x,x) = (x'x + c)^d
            #for d = 2 and c = self.kernel_arg
            #k(x,x) = <xn^2,.. ,x1^2, sqrt(2)*xn*xn-1, ..., sqrt(2)*xn*xn1, sqrt(2)*xn-1*xn-2, ..., sqrt(2)*xn-1*x1..., sqrt(2)*x2*x1, ..., sqrt(2c)*xn, ..., sqrt(2c)*x1, c>
            Z = [np.ones([X.shape[0],])*(self.kernel_arg**2)] #c

            for i in range(X.shape[1]):
                Z = np.vstack([Z, np.power(X[:,i],2)]) # xn^2
                Z = np.vstack([Z, X[:,i] * np.sqrt(2*self.kernel_arg)]) #sqrt(2c)*xn

            for i in range(1,X.shape[1]):
                for j in range(0,i):
                    Z = np.vstack([Z, np.sqrt(2) * np.multiply(X[:,i], X[:,j])]) #sqrt(2)*xn*xn1

            return Z.T

        raise NotImplementedError("transforming by kernel " + self.kernel)

    def fit(self, X, Y, **kwargs):
        Y=Y*2-1
        Z = self.transform(X)
        w = cvx.Variable([Z.shape[1], 1])

        if len(Y.shape) == 1:
            Y = np.reshape(Y, (Y.shape[0], 1))

        #https://www.cvxpy.org/tutorial/functions/index.html
        #classification loss functions
        if self.loss == 'logistic':
            obj = cvx.Minimize(cvx.sum(cvx.logistic(-cvx.multiply(Y,Z@w)))) # cvx.logistic(x) = log(1 + e^x)
        elif self.loss == 'soft_margin':
            obj = cvx.Minimize(cvx.sum(cvx.pos(1-cvx.multiply(Y,Z@w)))) # cvx.pos(x) = max{0,x}
        elif self.loss == 'quadratic_soft_margin':
            obj = cvx.Minimize(cvx.sum(cvx.power(cvx.pos(1-cvx.multiply(Y,Z@w)), 2)))
        else:
            raise NotImplementedError("fitting for " + self.loss + " as loss function")

        prob = cvx.Problem(obj).solve(verbose=True)
        self.params = w.value

    def predict(self, X, output_as_classifier=True, **kwargs):
        if self.params is None:
            raise UnboundLocalError("it is not possible to predict the data without training")

        Z = self.transform(X)
        YL = Z@self.params

        if output_as_classifier:
            YL = np.sign(YL)

        YL = (YL+1)/2
        return YL


# class KernelClassifier:
#     def __init__(self, verbose = 2, kernel='polynomial', loss='logistic', kernel_arg=1, loss_arg=1):
#         self.trained = False
#         self.verbose = verbose
#         self.kernel = kernel
#         self.loss = loss
#         self.params = None
#         self.kernel_arg = kernel_arg
#         self.loss_arg = loss_arg
#         self.convert_output = False
#         self.ravel_output = False

#     def __str__(self):
#         m_str = 'Class KernelClassifier\n'
#         if self.trained:
#             m_str += 'Model is fitted, \n'
#         else:
#             m_str += 'Model is not fitted, \n'
#         m_str += "Model created with kernel " + self.kernel + " and " + self.loss + " as loss function"
#         return m_str

#     def transform(self, X):
#         if self.kernel == 'polynomial':
#             #k(x,x) = (x'x + c)^d
#             #for d = 2 and c = self.kernel_arg
#             #k(x,x) = <xn^2,.. ,x1^2, sqrt(2)*xn*xn-1, ..., sqrt(2)*xn*xn1, sqrt(2)*xn-1*xn-2, ..., sqrt(2)*xn-1*x1..., sqrt(2)*x2*x1, ..., sqrt(2c)*xn, ..., sqrt(2c)*x1, c>
#             Z = [np.ones([X.shape[0],])*(self.kernel_arg**2)] #c

#             for i in range(X.shape[1]):
#                 Z = np.vstack([Z, np.power(X[:,i],2)]) # xn^2
#                 Z = np.vstack([Z, X[:,i] * np.sqrt(2*self.kernel_arg)]) #sqrt(2c)*xn

#             for i in range(1,X.shape[1]):
#                 for j in range(0,i):
#                     Z = np.vstack([Z, np.sqrt(2) * np.multiply(X[:,i], X[:,j])]) #sqrt(2)*xn*xn1

#             return Z.T

#         raise NotImplementedError("transforming by kernel " + self.kernel)

#     def fit(self, X, Y, trn_id=None, val_id=None, random_state=0):
#         if trn_id is not None:
#             X = X[trn_id]
#             Y = Y[trn_id]

#         if min(Y) == 0: # regressions operating with classes -1 and 1 not 0 and 1
#             Y=Y*2-1
#             self.convert_output = True

#         Z = self.transform(X)
#         w = cvx.Variable([Z.shape[1], 1])

#         if len(Y.shape) == 1: # input format (n_samples,) required in format (n_samples,1)
#             Y = np.reshape(Y, (Y.shape[0], 1))
#             self.ravel_output = True


#         #https://www.cvxpy.org/tutorial/functions/index.html
#         #classification loss functions
#         if self.loss == 'logistic':
#             obj = cvx.Minimize(cvx.sum(cvx.logistic(-cvx.multiply(Y,Z@w)))) # cvx.logistic(x) = log(1 + e^x)
#         elif self.loss == 'soft_margin':
#             obj = cvx.Minimize(cvx.sum(cvx.pos(1-cvx.multiply(Y,Z@w)))) # cvx.pos(x) = max{0,x}
#         elif self.loss == 'quadratic_soft_margin':
#             obj = cvx.Minimize(cvx.sum(cvx.power(cvx.pos(1-cvx.multiply(Y,Z@w)), 2)))
#         #regression loss functions
#         elif self.loss == 'squared_loss':
#             obj = cvx.Minimize(cvx.sum(cvx.power(Y-Z@w, 2)))
#         elif self.loss == 'e-insensitive':
#             obj = cvx.Minimize(cvx.sum(cvx.pos(cvx.abs(Y-Z@w) - self.loss_arg)))
#         elif self.loss == 'huber':
#             obj = cvx.Minimize(cvx.sum(cvx.huber(Y-Z@w, self.loss_arg)))
#         else:
#             raise NotImplementedError("fitting for " + self.loss + " as loss function")

#         prob = cvx.Problem(obj).solve()
#         self.params = w.value
#         self.trained = True
        
#     def predict(self, X):
#         if not self.trained:
#             raise UnboundLocalError("it is not possible to predict the data without training")

#         if self.kernel == 'polynomial':
#             Z = self.transform(X)
#             YL = np.sign(Z@self.params)
#             if self.convert_output:
#                 YL = (YL+1)/2

#             if self.ravel_output:
#                 return YL.ravel()
#             return YL

#         raise NotImplementedError("kernel predicting by  " + self.kernel)
    
#     def save(self, file_path):
#         with open(file_path,'wb') as file_handler:
#             joblib.dump([self.trained, self.verbose, self.kernel, self.loss, self.params, self.kernel_arg, self.loss_arg, self.convert_output, self.ravel_output], file_handler)

#     def load(self, file_path):
#         with open(file_path,'rb') as file_handler:
#             [self.trained, self.verbose, self.kernel, self.loss, self.params, self.kernel_arg, self.loss_arg, self.convert_output, self.ravel_output]= joblib.load(file_handler)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class geiger:
    def __init__(self,iterasi_num, vp, x, y, z, t):
        self.iter = iterasi_num
        self.vp = vp
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        
        self.res=[]

        self.ttx=[]
        self.tty=[]
        self.ttz=[]

        self.ttxttx=[]
        self.ttytty=[]
        self.ttzttz=[]

        self.ttxtty=[]
        self.ttxttz=[]
        self.ttyttz=[]

        self.ttxres=[]
        self.ttyres=[]
        self.ttzres=[]
    
    def cal(self):
        jarak=[]
        for i in range(len(self.x)-1):
            _jarak = np.sqrt((self.x[0]-self.x[i+1])**2+(self.y[0]-self.y[i+1])**2+(self.z[0]-self.z[i+1])**2)
            jarak.append(_jarak)
            self.res.append(self.t[i+1]-(jarak[i]/self.vp+self.t[0]))
            
            self.ttx.append((self.x[0]-self.x[i+1])/self.vp*jarak[i])
            self.tty.append((self.y[0]-self.y[i+1])/self.vp*jarak[i])
            self.ttz.append((self.z[0]-self.z[i+1])/self.vp*jarak[i])
            
            self.ttxttx.append(self.ttx[i]**2)
            self.ttytty.append(self.tty[i]**2)
            self.ttzttz.append(self.ttz[i]**2)
        
            self.ttxtty.append(self.ttx[i]*self.tty[i])
            self.ttxttz.append(self.ttx[i]*self.ttz[i])
            self.ttyttz.append(self.tty[i]*self.ttz[i])
            
            self.ttxres.append(self.ttx[i]*self.res[i])
            self.ttyres.append(self.tty[i]*self.res[i])
            self.ttzres.append(self.ttz[i]*self.res[i])
        return(jarak)
    
    def jacobian(self):
        self.cal()
        self.Jacob = np.array([[np.sum(self.ttxttx),np.sum(self.ttxtty),np.sum(self.ttxttz),np.sum(self.ttx)],
                        [np.sum(self.ttxtty),np.sum(self.ttytty),np.sum(self.ttyttz),np.sum(self.tty)],
                        [np.sum(self.ttxttz),np.sum(self.ttyttz),np.sum(self.ttzttz),np.sum(self.ttz)],
                        [np.sum(self.ttx),np.sum(self.tty),np.sum(self.ttz),6]])
        return self.Jacob
    
    def inversi_res(self):
        J = self.jacobian()
        Y = np.array([[np.sum(self.ttxres)],[np.sum(self.ttyres)],[np.sum(self.ttzres)],[np.sum(self.res)]])
        J_inv = np.linalg.inv(J) #invers matrix J
        X = J_inv.dot(Y)#perkalian matrix
        
        dx=X[0]
        dy=X[1]
        dz=X[2]
        dt=X[3]
        
        x0= self.x[0]+dx #△x
        y0= self.y[0]+dy #△y
        z0= self.z[0]+dz #△z
        t0= self.t[0]+dt #△t
        
        return x0,y0,z0,t0
    
    def rms(self):
        _rms = 0
        for i in range(len(self.x)-1):  # noqa: E999
            v = self.res[i]**2
            _rms += v
        rms = np.sqrt(_rms/len(self.res))
        return rms
    
    def update_values(self, new_x, new_y, new_z, new_t):
        self.x[0] = new_x.item()
        self.y[0] = new_y.item()
        self.z[0] = new_z.item()
        self.t[0] = new_t.item()
        
    def detik_ke_utc(self, detik):
        jam = detik // 3600
        menit = (detik % 3600) // 60
        detik = detik % 60
        return f'{int(jam):02}:{int(menit):02}:{detik:0.4f} UTC'
        
    def iterasi(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.2)
        fig1.subplots_adjust(wspace=0.3)
        axes[0].scatter(self.x[1:],self.y[1:], color="darkblue", marker='o',cmap='viridis', label='Stasiun')
        axes[0].scatter(self.x[0],self.y[0], color="red", marker='o',cmap='viridis', label='Stasiun')
        axes[0].grid(True)
        axes[0].set_title("Plot Stasiun dan Episenter sebelum")
        
        x_ = [self.x[0]]
        y_ = [self.y[0]]
        z_ = [-self.z[0]]
        
        if self.iter == 1:
            x,y,z,t = self.inversi_res()
            self.update_values(x,y,z,t)
            print(x,y,z,self.detik_ke_utc(37680+t.item()))
            x_.append(x)
            y_.append(y)
            z_.append(-(z.item()))
            axes[1].scatter(self.x[1:],self.y[1:], color="darkblue", marker='o',cmap='viridis', label='Stasiun')
            axes[1].scatter(self.x[0],self.y[0], color="red", marker='o',cmap='viridis', label='Stasiun')
            axes[1].set_title("Plot Stasiun dan Episenter setelah")
            print(self.rms())
            axes[1].grid(True)
            axes[2].plot(range(self.iter),self.rms())
            axes[2].scatter(range(self.iter),self.rms(), color="red", marker='o',cmap='viridis', label='Stasiun')
            axes[2].set_xlabel("Iterasi")
            axes[2].set_ylabel("RMS")
            axes[2].set_title("Plot RMS vs Iterasi")
        if self.iter > 1:
            rms = []
            for i in range(self.iter):
                x,y,z,t = self.inversi_res()
                self.update_values(x,y,z,t)
                y_.append(y.item())
                x_.append(x.item())
                z_.append(-(z.item()))
                print(self.rms())
                rms.append(self.rms())
            print(x,y,z,self.detik_ke_utc(37680+t.item()))
            axes[1].scatter(self.x[1:],self.y[1:], color="darkblue", marker='o',cmap='viridis', label='Stasiun')
            axes[1].scatter(self.x[0],self.y[0], color="red", marker='o',cmap='viridis', label='Stasiun')
            axes[1].grid(True)
            axes[1].set_title("Plot Stasiun dan Episenter setelah")
            axes[2].plot(range(self.iter),rms)
            axes[2].scatter(range(self.iter),rms, color="red", marker='o',cmap='viridis', label='Stasiun')
            axes[2].set_xlabel("Iterasi")
            axes[2].set_ylabel("RMS")
            axes[2].set_title("Plot RMS vs Iterasi")

        axes1[0].plot(x_,y_)
        axes1[0].set_title("X vs Y")
        axes1[1].plot(x_,z_)
        axes1[1].set_title("X vs Z")
        axes1[2].plot(y_,z_)
        axes1[2].set_title("Y vs Z")
        axes1[0].scatter(x_,y_, color="blue", marker='o',cmap='viridis', label='Stasiun')
        axes1[0].set_xlabel("X")
        axes1[0].set_ylabel("Y")
        axes1[1].scatter(x_,z_, color="blue", marker='o',cmap='viridis', label='Stasiun')
        axes1[1].set_ylabel("Z")
        axes1[1].set_xlabel("X")
        axes1[2].scatter(y_,z_, color="blue", marker='o',cmap='viridis', label='Stasiun')
        axes1[2].set_xlabel("Y")
        axes1[2].set_ylabel("Z")
        x_ = np.array(x_)
        y_ = np.array(y_)
        z_ = np.array(z_)

        # Then proceed with plotting
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(x_, y_, z_, marker='o')
        ax3d.scatter(0, 0, 0, marker='o',alpha=0.0)
            

        
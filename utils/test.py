import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plottensorO(Sxx, Syy, Szz, alpha, beta, gamma, xc, yc, zc, showPAS, showPASlabels, axislabels, POScolor, NEGcolor, PAScolor):
    Sxxstart, Syystart, Szzstart = -np.abs(Sxx) - sAoff, -np.abs(Syy) - sAoff, -np.abs(Szz) - sAoff if PASLong else -0 * sAoff, -0 * sAoff, -0 * sAoff

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if showPAS and showPASlabels:
        ax.plot_surface(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), color=PAScolor)  # Example surface plot, you need to replace it with your actual data
        ax.text(Arot(alpha, beta, gamma).dot(np.array([0, 0, np.abs(Szz) + sLoff])) + np.array([xc, yc, zc]), axislabels[2], fontsize=fontsize)
        ax.text(Arot(alpha, beta, gamma).dot(np.array([0, np.abs(Syy) + sLoff, 0])) + np.array([xc, yc, zc]), axislabels[1], fontsize=fontsize)
        ax.text(Arot(alpha, beta, gamma).dot(np.array([np.abs(Sxx) + sLoff, 0, 0])) + np.array([xc, yc, zc]), axislabels[0], fontsize=fontsize)

        th = np.linspace(0, np.pi, 100)
        ph = np.linspace(0, 2 * np.pi, 100)
        TH, PH = np.meshgrid(th, ph)
        X = (pos(rovaloid(Sxx, Syy, Szz, alpha, beta, gamma, TH, PH)) * np.sin(TH) * np.cos(PH) + xc)
        Y = (pos(rovaloid(Sxx, Syy, Szz, alpha, beta, gamma, TH, PH)) * np.sin(TH) * np.sin(PH) + yc)
        Z = (pos(rovaloid(Sxx, Syy, Szz, alpha, beta, gamma, TH, PH)) * np.cos(TH) + zc)
        ax.plot_surface(X, Y, Z, color=POScolor, alpha=opacity, rstride=1, cstride=1, linewidth=0, antialiased=False)

        ax.set_box_aspect([1,1,1])  # Aspect ratio is not straightforward, may need to adjust

    if showPAS:
        ax.plot_surface(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), color=PAScolor)  # Example surface plot, you need to replace it with your actual data

        th = np.linspace(0, np.pi, 100)
        ph = np.linspace(0, 2 * np.pi, 100)
        TH, PH = np.meshgrid(th, ph)
        X = (pos(rovaloid(Sxx, Syy, Szz, alpha, beta, gamma, TH, PH)) * np.sin(TH) * np.cos(PH) + xc)
        Y = (pos(rovaloid(Sxx, Syy, Szz, alpha, beta, gamma, TH, PH)) * np.sin(TH) * np.sin(PH) + yc)
        Z = (pos(rovaloid(Sxx, Syy, Szz, alpha, beta, gamma, TH, PH)) * np.cos(TH) + zc)
        ax.plot_surface(X, Y, Z, color=POScolor, alpha=opacity, rstride=1, cstride=1, linewidth=0, antialiased=False)

        ax.set_box_aspect([1,1,1])  # Aspect ratio is not straightforward, may need to adjust

    plt.show()

# You need to define the following functions: Arot, pos, rovaloid


def plottensorOplanes(sxx, syy, szz, a, b, g, xc, yc, zc, POScolor, NEGcolor, thick):
    numpts = 100
    phstep = 2 * np.pi / numpts
    thstep = 2 * np.pi / numpts
    
    # XY Plane
    th = np.pi / 2
    ph_values = np.linspace(0, 2 * np.pi, numpts)
    
    pts1 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      Arot(a, b, g) @ ([0, 0, -thick/2] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc]]
                     for ph in ph_values])
    
    pts1XYp = pts1[pts1[:, 0] >= 0][:, 1]
    pts1XYn = pts1[pts1[:, 0] <= 0][:, 1]
    
    pts2 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      Arot(a, b, g) @ ([0, 0, thick/2] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc]]
                     for ph in ph_values])
    
    pts2XYp = pts2[pts2[:, 0] >= 0][:, 1]
    pts2XYn = pts2[pts2[:, 0] <= 0][:, 1]
    
    pts3 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      [Arot(a, b, g) @ ([0, 0, -thick/2] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([0, 0, thick/2] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([0, 0, thick/2] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph + phstep) * [np.sin(th + thstep) * np.cos(ph + phstep), np.sin(th + thstep) * np.sin(ph + phstep), np.cos(th + thstep)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([0, 0, -thick/2] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph + phstep) * [np.sin(th + thstep) * np.cos(ph + phstep), np.sin(th + thstep) * np.sin(ph + phstep), np.cos(th + thstep)]) + [xc, yc, zc]]]
                     for ph in ph_values])
    
    pts3XYp = pts3[pts3[:, 0] >= 0][:, 1]
    pts3XYn = pts3[pts3[:, 0] <= 0][:, 1]
    
    # XZ Plane
    th_values = np.linspace(0, 2 * np.pi, numpts)
    
    pts1 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      Arot(a, b, g) @ ([0, -thick/2, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc]]
                     for th in th_values])
    
    pts1XZp = pts1[pts1[:, 0] >= 0][:, 1]
    pts1XZn = pts1[pts1[:, 0] <= 0][:, 1]
    
    pts2 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      Arot(a, b, g) @ ([0, thick/2, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc]]
                     for th in th_values])
    
    pts2XZp = pts2[pts2[:, 0] >= 0][:, 1]
    pts2XZn = pts2[pts2[:, 0] <= 0][:, 1]
    
    pts3 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      [Arot(a, b, g) @ ([0, -thick/2, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([0, thick/2, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([0, thick/2, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th + thstep, ph) * [np.sin(th + thstep) * np.cos(ph), np.sin(th + thstep) * np.sin(ph), np.cos(th + thstep)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([0, -thick/2, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th + thstep, ph) * [np.sin(th + thstep) * np.cos(ph), np.sin(th + thstep) * np.sin(ph), np.cos(th + thstep)]) + [xc, yc, zc]]]
                     for th in th_values])
    
    pts3XZp = pts3[pts3[:, 0] >= 0][:, 1]
    pts3XZn = pts3[pts3[:, 0] <= 0][:, 1]
    
    # YZ Plane
    ph = np.pi / 2
    
    pts1 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      Arot(a, b, g) @ ([-thick/2, 0, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc]]
                     for th in th_values])
    
    pts1YZp = pts1[pts1[:, 0] >= 0][:, 1]
    pts1YZn = pts1[pts1[:, 0] <= 0][:, 1]
    
    pts2 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      Arot(a, b, g) @ ([thick/2, 0, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc]]
                     for th in th_values])
    
    pts2YZp = pts2[pts2[:, 0] >= 0][:, 1]
    pts2YZn = pts2[pts2[:, 0] <= 0][:, 1]
    
    pts3 = np.array([[rovaloid(sxx, syy, szz, 0, 0, 0, th, ph),
                      [Arot(a, b, g) @ ([-thick/2, 0, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([thick/2, 0, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th, ph) * [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([thick/2, 0, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th + thstep, ph) * [np.sin(th + thstep) * np.cos(ph), np.sin(th + thstep) * np.sin(ph), np.cos(th + thstep)]) + [xc, yc, zc],
                       Arot(a, b, g) @ ([-thick/2, 0, 0] + rovaloid(sxx, syy, szz, 0, 0, 0, th + thstep, ph) * [np.sin(th + thstep) * np.cos(ph), np.sin(th + thstep) * np.sin(ph), np.cos(th + thstep)]) + [xc, yc, zc]]]
                     for th in th_values])
    
    pts3YZp = pts3[pts3[:, 0] >= 0][:, 1]
    pts3YZn = pts3[pts3[:, 0] <= 0][:, 1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(pts3XYp[:, 0], pts3XYp[:, 1], pts3XYp[:, 2], color=POScolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(np.concatenate((pts1XYp[:, 0], pts2XYp[:, 0])), np.concatenate((pts1XYp[:, 1], pts2XYp[:, 1])), np.concatenate((pts1XYp[:, 2], pts2XYp[:, 2])), color=POScolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(pts3XYn[:, 0], pts3XYn[:, 1], pts3XYn[:, 2], color=NEGcolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(np.concatenate((pts1XYn[:, 0], pts2XYn[:, 0])), np.concatenate((pts1XYn[:, 1], pts2XYn[:, 1])), np.concatenate((pts1XYn[:, 2], pts2XYn[:, 2])), color=NEGcolor, alpha=1, linewidth=0, antialiased=True)
    
    ax.plot_trisurf(pts3XZp[:, 0], pts3XZp[:, 1], pts3XZp[:, 2], color=POScolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(np.concatenate((pts1XZp[:, 0], pts2XZp[:, 0])), np.concatenate((pts1XZp[:, 1], pts2XZp[:, 1])), np.concatenate((pts1XZp[:, 2], pts2XZp[:, 2])), color=POScolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(pts3XZn[:, 0], pts3XZn[:, 1], pts3XZn[:, 2], color=NEGcolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(np.concatenate((pts1XZn[:, 0], pts2XZn[:, 0])), np.concatenate((pts1XZn[:, 1], pts2XZn[:, 1])), np.concatenate((pts1XZn[:, 2], pts2XZn[:, 2])), color=NEGcolor, alpha=1, linewidth=0, antialiased=True)
    
    ax.plot_trisurf(pts3YZp[:, 0], pts3YZp[:, 1], pts3YZp[:, 2], color=POScolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(np.concatenate((pts1YZp[:, 0], pts2YZp[:, 0])), np.concatenate((pts1YZp[:, 1], pts2YZp[:, 1])), np.concatenate((pts1YZp[:, 2], pts2YZp[:, 2])), color=POScolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(pts3YZn[:, 0], pts3YZn[:, 1], pts3YZn[:, 2], color=NEGcolor, alpha=1, linewidth=0, antialiased=True)
    ax.plot_trisurf(np.concatenate((pts1YZn[:, 0], pts2YZn[:, 0])), np.concatenate((pts1YZn[:, 1], pts2YZn[:, 1])), np.concatenate((pts1YZn[:, 2], pts2YZn[:, 2])), color=NEGcolor, alpha=1, linewidth=0, antialiased=True)
    
    ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1
    ax.set_axis_off()  # Turn off the axis
    plt.show()

# Define rovaloid and Arot functions accordingly

# Example usage
plottensorOplanes(sxx=1, syy=1, szz=1, a=1, b=1, g=1, xc=0, yc=0, zc=0, POScolor='blue', NEGcolor='red', thick=0.1)

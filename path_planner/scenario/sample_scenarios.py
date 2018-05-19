import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from rospace_lib import CartesianTEME, CartesianLVLH, QNSRelOrbElements, KepOrbElem


mean = np.array([0, 0, 0, 0, 0, 0])

sigmab = np.array([[0.4**2, 0, 0, 0, 0, 0],
          [0, 0.7**2, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0.7**2, 0, 0],
          [0, 0, 0, 0, 0.5**2, 0],
          [0, 0, 0, 0, 0, 0.1**2]])

sigmaest = np.array([[0.0093, 0.0116, 0.0043, -0.0086, -0.0106, 0.0008],
          [0.0116, 0.1215, 0.0022, -0.1296, -0.0120, -0.0091],
          [0.0043, 0.0022, 0.0118, 0.0025, -0.0063, -0.0033],
          [-0.0086, -0.1296, 0.0025, 0.2113, 0.0082, -0.0054],
          [-0.0106, -0.0120, -0.0063, 0.0082, 0.0126, 0.0010],
          [0.0008, -0.0091, -0.0033, -0.0054, 0.0010, 0.0734]])

cov = sigmab + 3 * sigmaest
print cov

x, y, z, vx, vy, vz = np.random.multivariate_normal(mean, cov, 5000).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)

chaserTEME = CartesianTEME()
chaserTEME.R = np.array([-443.1013869, 2100.66119018, -6755.7092574])
chaserTEME.V = np.array([4.83331984, -5.37836685, -1.98628992])

targetTEME = CartesianTEME()
targetTEME.R = np.array([-451.57095751, 2109.3595738, -6749.06351459])
targetTEME.V = np.array([4.83263844, -5.37804409, -2.00124123])

chaserLVLH = CartesianLVLH()
chaserLVLH.from_cartesian_pair(chaserTEME, targetTEME)

chaserKep = KepOrbElem()
chaserKep.from_cartesian(chaserTEME)

targetKep = KepOrbElem()
targetKep.from_cartesian(targetTEME)

sampleLVLH = CartesianLVLH()
sampleTEME = CartesianTEME()
true_roe = QNSRelOrbElements()
scaled_roe = QNSRelOrbElements()
roe = QNSRelOrbElements()
roe.from_keporb(targetKep, chaserKep)

samples = []

realChaserLVLH = CartesianLVLH()

for i in xrange(0, 5000):
    sampleLVLH.R = - np.array([x[i], y[i], z[i]])
    sampleLVLH.V = - 1e-3 * np.array([vx[i], vy[i], vz[i]])

    sampleTEME.from_lvlh_frame(targetTEME, sampleLVLH)

    realChaserLVLH.from_cartesian_pair(chaserTEME, sampleTEME)

    samples.append((sampleTEME, np.array([x[i], y[i], z[i], vx[i], vy[i], vz[i]])))

P = np.array([[0.0,0.0,0.0,0.0,0.0,0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

n = 0
for j in xrange(0, 25):
    nr = int(np.random.random() * len(samples))

    mahalo = np.sqrt((samples[nr][1] - mean).dot(np.linalg.inv(cov).dot(samples[nr][1] - mean)))

    targetKep.from_cartesian(samples[nr][0])

    true_roe.from_keporb(targetKep, chaserKep)
    scaled_roe = true_roe.as_scaled(chaserKep.a)

    file = open(str(nr) + '.scenario', 'w')

    file.write('chaser_state_vect:')
    file.write('  Rx:')
    file.write('  - ' + str(chaserTEME.R[0]))
    file.write('  - ' + str(chaserTEME.R[1]))
    file.write('  - ' + str(chaserTEME.R[2]))
    file.write('  V:')
    file.write('  - ' + str(chaserTEME.V[0]))
    file.write('  - ' + str(chaserTEME.V[1]))
    file.write('  - ' + str(chaserTEME.V[2]))
    file.write('mahalonobis_dist: ' + str(mahalo))
    file.write('roe:')
    file.write('  dA: -2.57576894464e-05')
    file.write('  dL: -0.0019361934494')
    file.write('  dIx: -8.93991572111e-05')
    file.write('  dIy: -0.000321171042324')
    file.write('  dEx: -0.000133263346855')
    file.write('  dEy: -0.000407535306936')
    file.write('roe_scaled:')
    file.write('  dA: -182.594057955')
    file.write('  dL: -13725.5097996')
    file.write('  dIx: -633.742980978')
    file.write('  dIy: -2276.75405581')
    file.write('  dEx: -944.692470555')
    file.write('  dEy: -2888.98294266')
    file.write('target_state_vect:')
    file.write('  Rx:')
    file.write('  - ' + str(samples[nr][0].R[0]))
    file.write('  - ' + str(samples[nr][0].R[1]))
    file.write('  - ' + str(samples[nr][0].R[2]))
    file.write('  V:')
    file.write('  - ' + str(samples[nr][0].V[0]))
    file.write('  - ' + str(samples[nr][0].V[1]))
    file.write('  - ' + str(samples[nr][0].V[2]))
    file.write('true_roe:')
    file.write('  dA: ' + str(true_roe.dA))
    file.write('  dL: ' + str(true_roe.dL))
    file.write('  dIx: ' + str(true_roe.dIx))
    file.write('  dIy: ' + str(true_roe.dIy))
    file.write('  dEx: ' + str(true_roe.dEx))
    file.write('  dEy: ' + str(true_roe.dEy))
    file.write('true_roe_scaled:')
    file.write('  dA: ' + str(scaled_roe.dA))
    file.write('  dL: ' + str(scaled_roe.dL))
    file.write('  dIx: ' + str(scaled_roe.dIx))
    file.write('  dIy: ' + str(scaled_roe.dIy))
    file.write('  dEx: ' + str(scaled_roe.dEx))
    file.write('  dEy: ' + str(scaled_roe.dEy))
    file.close()

    P += np.outer(true_roe.as_vector() - roe.as_vector(), true_roe.as_vector() - roe.as_vector())
    n += 1.0

P *= 1 / (n - 1)
file = open('sample_covariance.txt', 'w')
for row in P:
    for col in row:
        file.write(str(col))

file.close()

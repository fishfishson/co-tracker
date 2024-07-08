import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd
from matplotlib.animation import PillowWriter
path = "Neuron 3d/" ################ location for 3d pkl file
fold_con = os.listdir("Neuron 3d")
f_name = []
angle_req = []
dataF = pd.DataFrame()
absolute = False
for p in fold_con:
  data = p ####################### Name for 3d pkl file
  f_name.append(p)
  with open(os.path.join(path, data), 'rb') as f:
    cell = pickle.load(f)

  rotation = [0]
  local_r = 0
  euler_angles = []
  abs_rotation = True ###### choice for absolute and relative
  abs_indicator = 1 if abs_rotation else -1

  # Following codes show the line chart of rotation
  # predefine a rotation_axis
  #TODO: it may appreciated that we draw angle linechart in three views, e.g. x, y, z :)
  rotation_axis = np.array([1, 0, 0])
  euler_3 = []
  for i, frame in enumerate(cell.frames):
    euler_2 = []
    # if i % 10 != 0: continue
    rotvec = R.from_matrix(frame.locale_r).as_rotvec(degrees=True)
    euler_angles.append(R.from_matrix(frame.locale_r).as_euler('zxy', degrees=True))
    euler_2.append(R.from_matrix(frame.locale_r).as_euler('zxy', degrees=True)[1])
    # if abs_rotation:
    #   local_r += np.abs(np.linalg.norm(
    #     R.from_matrix(frame.locale_r).as_rotvec(degrees=True)))
    # else:
    #   local_r = np.linalg.norm(
    #     R.from_matrix(frame.r).as_rotvec(degrees=True))
    if abs_rotation:
      local_r += np.abs(np.linalg.norm(
        R.from_matrix(frame.locale_r).as_rotvec(degrees=True)))
    else:
      if rotation_axis @ rotvec > 0:
        local_r += np.linalg.norm(
          R.from_matrix(frame.locale_r).as_rotvec(degrees=True))
      else:
        local_r -= np.linalg.norm(
          R.from_matrix(frame.locale_r).as_rotvec(degrees=True))
    rotation.append(local_r)
    euler_3.append(euler_2[0])
  print(euler_3)
  dataF[p] = euler_3
  x_axis = np.arange(len(rotation)) * 339 #TODO:
  # print(x_axis, '\n',rotation)
  import pandas
  df = pandas.DataFrame(x_axis, rotation)
  # save_path = rf'Z:\Data to ZHANG Yi\figure_modification\{data}'
  # import os
  # if not os.path.exists(save_path):
  #   os.mkdir(save_path)
  # plt.plot(x_axis, rotation)
  # if abs_rotation:
  #   df.to_excel(rf'{path}\abs.xlsx')
  #   plt.draw()
  #   plt.savefig(rf"{path}\abs.png", dpi=400)
  # else:
  #   df.to_excel(rf'{path}\relevent.xlsx')
  #   plt.savefig(rf'{path}\relevent.png', dpi=400)
  # pandas.DataFrame(euler_angles).to_excel(rf'{path}\euler_angle.xlsx')
  # print(p)
  # print(rotation[60])


#
#
# dataF.to_csv("30% hydrogel 3D_y_axis.csv")
plt.show()
#
# following codes are to visualize point traces.
# You specify the point index you would like to visualize
# point_index = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14]
point_index = [2]   ####### The point tracked in 2d file
# Set to Ture if you want to draw the rotation axis trace instead of the point
vis_rotation_axis = False ##############
# Set to Ture if you want to draw trace with translation information
with_translation = False ############## Linear motion calculation
if vis_rotation_axis:
  point_index = [0]

matplotlib.use('tkagg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
l,= plt.plot([], [], [], 'k-')
line = []
rotation = []
for i, frame in enumerate(cell.frames):
  # if i % 1 != 0: continue
  center = frame.center
  if vis_rotation_axis:
    x_T = np.transpose(frame.r * frame.radius * 2)
  else:
    # x_T = np.transpose([frame.x, frame.y, frame.z], axes=(1,0))
    x_Tni  = np.array([frame.x, frame.y, frame.z])
  for j, index in enumerate(point_index):
    if len(rotation) <= j:
      rotation.append([])
    if with_translation:
      # rotation[j].append(
      #   [x_T[index][0] + center[0], x_T[index][1] + center[1], x_T[index][2]])
      rotation[j].append(
        [frame.x[index], frame.y[index], frame.z[index]])
    else:
      rotation[j].append(
        [x_T[0][index] - center[0], x_T[1][index] - center[1], x_T[2][index]])
      # rotation[j].append(
      #   [frame.x[index] - center[0], frame.y[index] - center[1], frame.z[index]])

flip_x = False
for i in range(len(rotation)):
  x, y, z = np.transpose(rotation[i]) * 0.065
  x -= ((np.max(x) + np.min(x)) / 2 )
  z -= np.min(z) - 1
  y -= ((np.max(y) + np.min(y)) / 2 )
  ax.set_xlim(-2.5, 2.5)
  ax.set_ylim(-2.5, 2.5)

  x= x[::-1]
  y = y[::-1]
  z = z[::-1]
  N = len(x)
  metadata = dict(title = "movie")
  writer =PillowWriter(fps = 7, metadata = metadata)
  with writer.saving(fig , "Neuron.gif", 300):
    for j in range(N - 1):
      if flip_x:
        ax.plot(-x[j:j + 2], y[j:j + 2], z[j:j + 2],
                color=plt.cm.jet((N-j) / N), alpha=1)
        ax.plot(-x[j:j + 2], y[j:j + 2], 0,
                color=plt.cm.gray((j-N)/ N), alpha=1)
      else:
        ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
                color=plt.cm.jet((j) / N), alpha=1, linewidth = 3)

        # ax.plot(x[j:j + 2], y[j:j + 2], 0,
        #         color=plt.cm.gray( (j) / N), alpha=1)
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(-3, 3)
        ax.set_zlim3d(0, 6)
        ax.view_init(60,-110)

        writer.grab_frame()
ax.set_xlabel('X', fontsize = 40)
ax.set_ylabel('Y', fontsize = 40)
ax.set_zlabel('Z', fontsize = 40)

ax.zaxis.set_tick_params(labelsize= 30)
ax.xaxis.set_tick_params(labelsize=30)
ax.yaxis.set_tick_params(labelsize=30)

plt.show()

fig = plt.gcf()
fig.set_dpi(300)
import matplotlib.animation as animation
def rotate(angle):
    ax.view_init(azim=angle)
def rotate2(angle):
    ax.view_init(elev=angle)

print("Making animation")
anim = animation.FuncAnimation(fig, rotate2, frames=np.arange(0, 362, 2), interval=100)

anim.save('rotation 3hr.gif', dpi=100, writer='imagemagick')




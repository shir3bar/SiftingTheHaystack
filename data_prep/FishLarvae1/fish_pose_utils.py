import numpy as np
import matplotlib.pyplot as plt

def calc_ellipse(centroid, semi_major_axis, semi_minor_axis, angle_minor_axis):
    # calc eye ellipse from vergence angle
    h, k = centroid
    a = semi_major_axis
    b = semi_minor_axis
    theta = np.deg2rad(angle_minor_axis)# Angle between the minor axis and the x-axis
    #print(angle_minor_axis)
    # Generate points along the ellipse
    t = np.linspace(0, 2 * np.pi, 1000)
    x = h + b * np.cos(t) * np.cos(theta) - a * np.sin(t) * np.sin(theta)
    y = k + b * np.cos(t) * np.sin(theta) + a * np.sin(t) * np.cos(theta)
    # Calculate the coordinates of major axis endpoints
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    major_endpoint_1 = R@np.array([0,a])+np.array([h,k])
    major_endpoint_2 = R@np.array([0,-a])+np.array([h,k])
    # Calculate the coordinates of minor axis endpoints
    minor_endpoint_1 = R@np.array([b,0])+np.array([h,k])
    minor_endpoint_2 = R@np.array([-b,0])+np.array([h,k])
    return x,y, np.stack([major_endpoint_1,major_endpoint_2,minor_endpoint_1,minor_endpoint_2])

def plot_fish_and_eyes(eyes,skel=None):
  # a is the semi-major axis, according to measurements taken from their videos the major axis is 30 pixels, making a=15
  # b is the semi-minor axis, according to measurements taken from their videos the minor axis is 20-22 pixels, let's make b=10.5
  # distance of eyes from head center 12-14 pixels on the x axis, 30 pixels on the y axis
    x_margin = 13
    y_margin = 30
    left_eye_centroid = (-x_margin, y_margin)
    right_eye_centroid = (x_margin, y_margin) 
    semi_major_axis = 15
    semi_minor_axis = 10.5
    x_left, y_left, axes_endpoints_left = calc_ellipse(left_eye_centroid, semi_major_axis, semi_minor_axis,180-eyes[0]) #remove 180 because vergence is 90-a and we're measuring counterclockwise from positive x-axis
    x_right, y_right, axes_endpoints_right = calc_ellipse(right_eye_centroid, semi_major_axis, semi_minor_axis,eyes[1]) #for right eye we don't need 180-

    # Plot the ellipse
    plt.figure()
    plt.plot(x_left,  y_left, label='Ellipse')
    plt.plot(x_right,  y_right, label='Ellipse')
    # Plot the centroid
    plt.plot(left_eye_centroid[0],left_eye_centroid[1], 'ro', label='Centroid')
    plt.plot(right_eye_centroid[0],right_eye_centroid[1], 'ro', label='Centroid')


    #x_major_endpoint_1 = h*np.cos(theta)+(k+a)*np.sin(theta)#h + a * np.cos(theta)
    #y_major_endpoint_1 = k + a * np.sin(theta)
    #x_major_endpoint_2 = h - a * np.cos(theta)
    #y_major_endpoint_2 = k - a * np.sin(theta)

    # Plot the major axis
    plt.plot(axes_endpoints_left[0:2,0],axes_endpoints_left[0:2,1] , 'g--', label='Major Axis')
    plt.plot(axes_endpoints_right[0:2,0],axes_endpoints_right[0:2,1] , 'g--', label='Major Axis')


    #x_minor_endpoint_1 = h - b * np.sin(theta)
    #y_minor_endpoint_1 = k + b * np.cos(theta)
    #x_minor_endpoint_2 = h + b * np.sin(theta)
    #y_minor_endpoint_2 = k - b * np.cos(theta)

    # Plot the minor axis
    plt.plot(axes_endpoints_left[2:,0],axes_endpoints_left[2:,1], 'b--', label='Minor Axis')
    plt.plot(axes_endpoints_right[2:,0],axes_endpoints_right[2:,1], 'b--', label='Minor Axis')

    if not skel is None:
      plt.plot(skel[:,0],skel[:,1],'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.show()


def get_exp_img_with_eyes(bout_sample,cluster_id,boutInd,save_dir=None,plot_ellipse=True):
    skeleton = bout_sample['bSkeleton']
    eyes = bout_sample['bEyes']
    opacities = np.linspace(0.1,1,skeleton.shape[2])
    figname = f'cluster_id_{cluster_id}_seq_id_{boutInd}'
    x_margin = 13
    y_margin = 30
    left_eye_centroid = (-x_margin, y_margin)
    right_eye_centroid = (x_margin, y_margin) 
    semi_major_axis = 15
    semi_minor_axis = 10.5
    fig = plt.figure(figsize=(5,10))
    for s in range(skeleton.shape[2]):
        skel = skeleton[:,:,s]
        curr_eyes = eyes[s,:]
        plt.scatter(skel[:,0],skel[:,1], alpha=opacities[s],color='steelblue')
        x_left, y_left, axes_endpoints_left = calc_ellipse(left_eye_centroid, semi_major_axis, semi_minor_axis,180-curr_eyes[0]) #remove 180 because vergence is 90-a and we're measuring counterclockwise from positive x-axis
        x_right, y_right, axes_endpoints_right = calc_ellipse(right_eye_centroid, semi_major_axis, semi_minor_axis,curr_eyes[1]) #for right eye we don't need 180-
        if plot_ellipse:
          plt.plot(x_left,  y_left, color='steelblue', alpha=opacities[s])
          plt.plot(x_right,  y_right, color='steelblue',alpha=opacities[s])
          cmd_major = 'g--'
          cmd_minor = '--'
        else:
          cmd_major = 'go'
          cmd_minor = 'o'
        plt.plot(axes_endpoints_left[0:2,0],axes_endpoints_left[0:2,1] , cmd_major, alpha=opacities[s])
        plt.plot(axes_endpoints_right[0:2,0],axes_endpoints_right[0:2,1] , cmd_major,  alpha=opacities[s])
        plt.plot(axes_endpoints_left[2:,0],axes_endpoints_left[2:,1], cmd_minor, color='orange', alpha=opacities[s])
        plt.plot(axes_endpoints_right[2:,0],axes_endpoints_right[2:,1], cmd_minor, color='orange', alpha=opacities[s])
        plt.plot(left_eye_centroid[0],left_eye_centroid[1], 'ro')
        plt.plot(right_eye_centroid[0],right_eye_centroid[1], 'ro')
    
    plt.title(figname)
    xmin,xmax,ymin,ymax = get_axes_lims(skeleton)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax+50))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    if not save_dir is None:
      plt.savefig(os.path.join(save_dir,f'{figname}.png'),
                bbox_inches='tight')
      plt.close()


def get_skeleton_with_eyes(bout_sample):
    skeleton = bout_sample['bSkeleton']
    eyes = bout_sample['bEyes']
    x_margin = 13
    y_margin = 30
    left_eye_centroid = (-x_margin, y_margin)
    right_eye_centroid = (x_margin, y_margin) 
    semi_major_axis = 15
    semi_minor_axis = 10.5
    left = []
    right = []
    new_skel = []
    for s in range(skeleton.shape[2]):
        skel = skeleton[:,:,s]
        curr_eyes = eyes[s,:]
        _,_, axes_endpoints_left = calc_ellipse(left_eye_centroid, semi_major_axis, semi_minor_axis,180-curr_eyes[0]) #remove 180 because vergence is 90-a and we're measuring counterclockwise from positive x-axis
        _,_, axes_endpoints_right = calc_ellipse(right_eye_centroid, semi_major_axis, semi_minor_axis,curr_eyes[1])
        new_skel.append(np.concatenate([axes_endpoints_left,axes_endpoints_right,skel]))
        #left.append(axes_endpoints_left)
        #right.append(axes_endpoints_right)
    new_skeleton = np.stack(new_skel,axis=2) 
    return new_skeleton

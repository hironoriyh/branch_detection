#!/usr/bin/env python  
import roslib
# roslib.load_manifest('learning_tf')
import rospy
import tf
# import turtlesim.msg

# 
# euler = tf.transformations.euler_from_quaternion(q)
# roll = euler[0]
# pitch = euler[1]
# yaw = euler[2]

# def handle_turtle_pose(msg, turtlename):
#     br = tf.TransformBroadcaster()
#     br.sendTransform((msg.x, msg.y, 0),
#                      tf.transformations.quaternion_from_euler(0, 0, msg.theta),
#                      rospy.Time.now(),
#                      turtlename,
#                      "world")
# 
# if __name__ == '__main__':
#     rospy.init_node('camera_frame_broadcaster')
#     turtlename = rospy.get_param('~turtle')
#     rospy.Subscriber('/%s/pose' % turtlename,
#                      turtlesim.msg.Pose,
#                      handle_turtle_pose,
#                      turtlename)
#     rospy.spin()

#  q = (qx, qy, qz, qw)


def q_mult(q1, q2):
    x1, y1, z1 ,w1 = q1
    x2, y2, z2 ,w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return x, y, z ,w

def q_inv(q):
    x, y, z ,w = q
    n = x*x + y*y +z*z + w*w
    return [-x/n, -y/n, -z/n, w/n]

# inner product
def q_iprod(q1,q2):
    q = [x*y for (x,y) in zip(q1,q2)] 
    return sum(q)
    
def q_conjugate(q):
    x, y, z ,w = q
    return [-x, -y, -z, w]

def q_normalize(q):
    x, y, z ,w = q
    n = math.sqrt(x*x + y*y +z*z + w*w)
    return [x/n, y/n, z/n, w/n]

# Linear interpolate with rate: [t,(1-t)] (with normalization)     
def q_LERP(q1,q2,t):
    q = [x*(1.0-t)+y*t for (x,y) in zip(q1,q2)]    
    return q_normalize(q)
    
# Sphere interpolate with rate: [t,(1-t)]   
def q_SLERP(q1,q2,t):
    w = math.acos(q_iprod(q1,q2))
    a = math.sin((1.0-t)*w)/ math.sin(w)
    b = math.sin(t*w)/ math.sin(w)
    q = [x*a+y*b for (x,y) in zip(q1,q2)]    
    return q

# Rotate vector using quaternion        
def qv_mult(q1, v1):
    q2 =  v1 + [0.0]
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[:3]


# q1 = [-0.048, -0.01, -0.7146, 0.697]
q1 = [0, 0, 0, 1]
new_quat = q_inv(q1)
print "inv quat", new_quat
quot_cameralink_to_depthframe = [-0.500, 0.500, -0.500, 0.500]
# quot_cameralink_to_depthframe = [-0.707, 0, 0, 0.707]
new_quat = q_mult(q1, quot_cameralink_to_depthframe)
print new_quat[0], new_quat[1], new_quat[2], new_quat[3] 

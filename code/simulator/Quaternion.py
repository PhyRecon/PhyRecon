import taichi as ti

# quaternion's operation
@ti.func
def quaternion_to_matrix(q):
    m00 = 1 - 2*q[2]*q[2] - 2*q[3]*q[3]
    m01 = 2 * (q[1]*q[2] - q[0]*q[3])
    m02 = 2 * (q[1]*q[3] + q[0]*q[2])
    
    m10 = 2 * (q[1]*q[2] + q[0]*q[3])
    m11 = 1 - 2*q[1]*q[1] - 2*q[3]*q[3]
    m12 = 2 * (q[2]*q[3] - q[0]*q[1])
    
    m20 = 2 * (q[1]*q[3] - q[0]*q[2])
    m21 = 2 * (q[2]*q[3] + q[0]*q[1])
    m22 = 1 - 2*q[1]*q[1] - 2*q[2]*q[2]
    
    return ti.Matrix([[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]])
    
    
@ti.func
def quaternion_multiplication(q1, q2):
    s1 = q1[0]
    v1 = ti.Vector([q1[1], q1[2], q1[3]])
    s2 = q2[0]
    v2 = ti.Vector([q2[1], q2[2], q2[3]])
    axis = s1*v2+s2*v1+ti.math.cross(v1, v2)
    angle = s1*s2-ti.math.dot(v1, v2)
    return ti.Vector([angle, axis[0], axis[1], axis[2]])
    
    
@ti.func
def quaternion_normalization(q):
    norm = ti.pow(ti.math.dot(q, q),-0.5)
    q = norm * q
    
    
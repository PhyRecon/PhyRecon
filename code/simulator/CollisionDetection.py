import taichi as ti
from simulator.RigidBody import skew_symmetric_matrix

proximity_range = 0.001
restitution_coef = 0.0
friction_coef = 0.8

@ti.kernel
def collision_detection(o1: ti.template(), o2: ti.template(), radius: ti.f64, dt: ti.f64):
    for i,j in ti.ndrange((1, o1.n_particles[0]),(1,o2.n_particles[0])):
            dist = ti.math.distance(o1.p_x[i], o2.p_x[j])
            if(dist<(radius*2)):               
                v1 = o1.r_v[0] + ti.math.cross(o1.r_w[0], o1.p_x[i]-o1.r_x[0])
                v2 = o2.r_v[0] + ti.math.cross(o2.r_w[0], o2.p_x[j]-o2.r_x[0])
                v = v1-v2
                N = ti.math.normalize(o1.p_x[i]-o2.p_x[j])
                vN_magnitude = ti.math.dot(v, N)
                vN =  vN_magnitude * N
                vT = v-vN
                alpha = ti.math.max( 1 - friction_coef*(1+restitution_coef)*ti.math.length(vN)/(ti.math.length(vT)+1e-6), 0.0)  
                if(vN_magnitude<-1e-5):           
                    desiredVN = -restitution_coef*vN
                    desiredVT = alpha*vT
                    desiredV = desiredVN + desiredVT
                
                    # update linear velocity
                    I_inv_1 = ti.math.inverse(o1.r_rotM[0] * o1.r_I[0] * o1.r_rotM[0].transpose())
                    I_inv_2 = ti.math.inverse(o2.r_rotM[0] * o2.r_I[0] * o2.r_rotM[0].transpose())

                    tmp1 = I_inv_1 @ ti.math.cross(o1.p_x[i]-o1.r_x[0], N)
                    tmp2 = I_inv_2 @ ti.math.cross(o2.p_x[j]-o2.r_x[0], N)
                    tmp11 = ti.math.dot(N, skew_symmetric_matrix(tmp1) @ (o1.p_x[i]-o1.r_x[0]))
                    tmp22 = ti.math.dot(N, skew_symmetric_matrix(tmp2) @ (o2.p_x[j]-o2.r_x[0]))
                    impulse = ti.math.length(desiredVN-vN)/(1.0/o1.r_mass[0] + 1.0/o2.r_mass[0] + tmp11 + tmp22)*N

                    o1.delta_v[0] +=  impulse/o1.r_mass[0]
                    o2.delta_v[0] += -impulse/o2.r_mass[0]
                    #print("velocity", o1.delta_v[0], o2.delta_v[0])

                    # update angular velocity
                    rotational_impulse_1 = ti.math.cross(o1.p_x[i]-o1.r_x[0], impulse)
                    rotational_impulse_2 = ti.math.cross(o2.p_x[j]-o2.r_x[0], impulse)
                    if(ti.abs(rotational_impulse_1[0])<1e-5):
                        rotational_impulse_1[0] = 0.0
                    if(ti.abs(rotational_impulse_1[1])<1e-5):
                        rotational_impulse_1[1] = 0.0
                    if(ti.abs(rotational_impulse_1[2])<1e-5):
                        rotational_impulse_1[2] = 0.0

                    if(ti.abs(rotational_impulse_2[0])<1e-5):
                        rotational_impulse_2[0] = 0.0
                    if(ti.abs(rotational_impulse_2[1])<1e-5):
                        rotational_impulse_2[1] = 0.0
                    if(ti.abs(rotational_impulse_2[2])<1e-5):
                        rotational_impulse_2[2] = 0.0
                    o1.delta_w[0] +=  I_inv_1 @ rotational_impulse_1
                    o2.delta_w[0] += -I_inv_2 @ rotational_impulse_2

                    o1.n_contact[0] += 1
                    o2.n_contact[0] += 1

                    if(-1*vN_magnitude*dt> proximity_range):
                        o1.sleep[0] = -1 
                        o2.sleep[0] = -1



@ti.kernel
def proximity_detection_with_fix_particles(o1: ti.template(), o2: ti.template(), radius: ti.f64, dt: ti.f64):
    for i,j in ti.ndrange((1, o1.n_particles[0]),(1,o2.n_particles[0])):
            dist = ti.math.distance(o1.p_x[i], o2.p_x[j])
            if(dist<(radius*2+proximity_range)):               
                v1 = o1.r_v[0] + ti.math.cross(o1.r_w[0], o1.p_x[i]-o1.r_x[0])
                v = v1
                N = ti.math.normalize(o1.p_x[i]-o2.p_x[j])
                vN_magnitude = ti.math.dot(v, N)
                vN =  vN_magnitude * N
                vT = v-vN
                alpha = ti.math.max( 1 - friction_coef*(1+restitution_coef)*ti.math.length(vN)/(ti.math.length(vT)+1e-6), 0.0)  
                if(vN_magnitude<-1e-5):
                    # if(o1.n_contact[0] >5500):       
                    desiredVN = -restitution_coef*vN
                    desiredVT = alpha*vT
                    desiredV = desiredVN + desiredVT
                
                    # update linear velocity
                    I_inv_1 = ti.math.inverse(o1.r_rotM[0] * o1.r_I[0] * o1.r_rotM[0].transpose())
                    vect1 = o1.p_x[i]-o1.r_x[0]
                    R_crossmatrix = ti.Matrix([[0, -vect1[2], vect1[1]], [vect1[2], 0, -vect1[0]], [-vect1[1], vect1[0], 0]])
                    identitymatrix = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    K = identitymatrix/o1.r_mass[0] - R_crossmatrix@I_inv_1@R_crossmatrix
                    impulse = ti.math.inverse(K)@(desiredV-v)

                    o1.delta_v[0] +=  impulse/o1.r_mass[0]

                    # update angular velocity
                    rotational_impulse_1 = ti.math.cross(o1.p_x[i]-o1.r_x[0], impulse)

                    o1.delta_w[0] +=  I_inv_1 @ rotational_impulse_1

                    o1.n_contact[0] += 1

                    if o1.p_x_collision[i][0] == 0:
                        # save the first collision point
                        o1.p_x_first_collision_pos[i] = o1.p_x[i]
                        o1.p_x_collision[i] = 1                         # mark the particle as collision

                    if(-1*vN_magnitude*dt> proximity_range):
                        o1.sleep[0] = -1 

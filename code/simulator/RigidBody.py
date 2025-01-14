import taichi as ti

from simulator.Quaternion import quaternion_to_matrix
from simulator.Quaternion import quaternion_multiplication
from simulator.Quaternion import quaternion_normalization

@ti.func
def skew_symmetric_matrix(q):
    m00 = 0.0
    m01 = -q[2]
    m02 = q[1]
    
    m10 = q[2]
    m11 = 0.0
    m12 = -q[0]
    
    m20 = -q[1]
    m21 = q[0]
    m22 = 0.0
    
    return ti.Matrix([[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]])
    

# physical properties for particle
@ti.data_oriented
class RigidBody:
    def __init__(self, dim, pmass, gravity, n_max):
        self.n_particles = ti.field(ti.i64,1)
        self.p_mass = pmass
        self.gravity = gravity
        self.p_x = ti.Vector.field(dim, ti.f64, n_max, needs_grad=True)
        self.p_x_pre = ti.Vector.field(dim, ti.f64, n_max, needs_grad=True)
        self.p_x_local = ti.Vector.field(dim, ti.f64, n_max, needs_grad=True)
        self.p_colors = ti.Vector.field(4, ti.f64, n_max)
        self.p_f = ti.Vector.field(dim, ti.f64, n_max)

        self.p_x_first_collision_pos = ti.Vector.field(dim, ti.f64, n_max)          # save the first collision position
        self.p_x_collision = ti.Vector.field(1, ti.f64, n_max)  # 0 for no collision, 1 for collision

        self.sleep = ti.field(ti.f64,1)
        self.sleep[0] = -1                              # -1: not sleep, 1: sleep
        self.r_mass = ti.field(ti.f64,1)
        self.r_I = ti.Matrix.field(dim, dim, ti.f64, 1)
       
        self.r_x = ti.Vector.field(dim, ti.f64, 1)
        self.r_v = ti.Vector.field(dim, ti.f64, 1)
        self.r_q = ti.Vector.field(4, ti.f64, 1)
        self.r_w = ti.Vector.field(3, ti.f64, 1)
        self.r_rotM = ti.Matrix.field(dim, dim, ti.f64, 1)

        self.r_x_pre = ti.Vector.field(dim, ti.f64, 1)
        self.r_v_pre = ti.Vector.field(dim, ti.f64, 1)
        self.r_q_pre = ti.Vector.field(4, ti.f64, 1)
        self.r_w_pre = ti.Vector.field(3, ti.f64, 1)
        
        self.n_contact = ti.field(ti.f64,1)
        self.delta_v = ti.Vector.field(dim, ti.f64, 1)
        self.delta_w = ti.Vector.field(dim, ti.f64, 1)
        self.max_distance = ti.field(ti.f64,1)
        self.rwa = ti.field(ti.f64,1) # ref: https://www10.cs.fau.de/publications/theses/2010/Schornbaum_DA_2010.pdf
        

    @ti.kernel
    def clear_all(self):

        self.n_particles[0] = 0

        self.p_x.fill(0.0)
        self.p_x_pre.fill(0.0)
        self.p_x_local.fill(0.0)
        self.p_colors.fill(0.0)
        self.p_f.fill(0.0)

        self.sleep[0] = -1

        self.r_x_pre.fill(0.0)
        self.r_v_pre.fill(0.0)
        self.r_q_pre.fill(0.0)
        self.r_w_pre.fill(0.0)

        self.delta_v.fill(0.0)
        self.delta_w.fill(0.0)

        print('clear all')
        

    @ti.kernel
    def set_init_rigid(self):
        self.r_mass[0] = self.p_mass * self.n_particles[0]
        self.r_I[0] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.r_x[0] = ti.Vector([0.0, 0.0, 0.0])
        self.r_q[0] = ti.Vector([1.0, 0.0, 0.0, 0.0]) 
        self.r_v[0] = ti.Vector([0.0,0.0,0.0])
        self.r_w[0] = ti.Vector([0.0,0.0,0.0])
        self.max_distance[0] = 0.0
        self.rwa[0] = 0.0
        self.r_rotM[0] = quaternion_to_matrix(self.r_q[0])

        self.p_x_first_collision_pos.fill(0.0)          # clear first collision position
        self.p_x_collision.fill(0.0)            # clear collision status


    @ti.kernel
    def set_rigid_property(self): 
        # compute center of mass
        for p in range(self.n_particles[0]):
            self.r_x[0] += self.p_x[p]*self.p_mass/self.r_mass[0]
        
        # compute relative position to CoM
        for t in range(self.n_particles[0]):
            self.p_x_local[t] = self.p_x[t] - self.r_x[0]
            ti.atomic_max(self.max_distance[0], ti.math.length(self.p_x_local[t]))

        # compute inertia matrix using relative pos
        for p in range(self.n_particles[0]):
            x2 = self.p_x_local[p][0]*self.p_x_local[p][0]
            y2 = self.p_x_local[p][1]*self.p_x_local[p][1]
            z2 = self.p_x_local[p][2]*self.p_x_local[p][2]
            r2 = x2 + y2 + z2
        
            self.r_I[0][0,0] += (r2 - x2)* self.p_mass
            self.r_I[0][0,1] -= self.p_mass * self.p_x_local[p][0] * self.p_x_local[p][1]
            self.r_I[0][0,2] -= self.p_mass * self.p_x_local[p][0] * self.p_x_local[p][2]
        
            self.r_I[0][1,0] -= self.p_mass * self.p_x_local[p][0] * self.p_x_local[p][1]
            self.r_I[0][1,1] += (r2 - y2)* self.p_mass
            self.r_I[0][1,2] -= self.p_mass * self.p_x_local[p][1] * self.p_x_local[p][2]
        
            self.r_I[0][2,0] -= self.p_mass * self.p_x_local[p][0] * self.p_x_local[p][2]
            self.r_I[0][2,1] -= self.p_mass * self.p_x_local[p][1] * self.p_x_local[p][2]
            self.r_I[0][2,2] += (r2 - z2)* self.p_mass


    # setup the initial velocity and angular velocity
    # wakeup rigid body for simulation
    @ti.kernel
    def set_init_state(self, initV: ti.template(), initW: ti.template()): 
        self.r_v[0] = initV
        self.r_w[0] = initW 
        self.sleep[0] = -1       # add setup sleep state in __init()__

    # compute particle position according to current rigid body configuration       
    @ti.kernel
    def update_particles_state(self):
        for p in range(self.n_particles[0]):
            self.p_x[p] =  self.r_rotM[0] @ self.p_x_local[p] + self.r_x[0]
            self.p_x_pre[p] = self.p_x[p]
            
    # before each simulation step
    # store last frame's configuration, including position, orientation, velocity and angular velocity
    # set the contact number, velocity and angular velocity change due to contact to zero                
    @ti.kernel
    def pre_simulation(self):
        self.r_x_pre[0] = self.r_x[0]
        self.r_v_pre[0] = self.r_v[0]
        self.r_q_pre[0] = self.r_q[0]
        self.r_w_pre[0] = self.r_w[0]
       

    # before each contact resolve step               
    @ti.kernel
    def pre_contact(self):
        self.delta_v[0] = ti.Vector([0.0,0.0,0.0])
        self.delta_w[0] = ti.Vector([0.0,0.0,0.0])
        self.n_contact[0] = 0

    # update rigid body state with only gravity taken into consideration
    @ti.kernel
    def compute_rigid_motion_wo_contact(self, dt:ti.f64):
        if(self.sleep[0]<0):
            # update linear velocity and position
            force = self.r_mass[0]*self.gravity
            self.r_v[0] +=  dt * (1/self.r_mass[0]) * force
            self.r_x[0] +=  dt * self.r_v[0]
    
            # update angular velocity and orientation
            angle = ti.pow(ti.math.dot(self.r_w[0], self.r_w[0]),0.5)
            axis = ti.Vector([0.0,1.0,0.0])
            if(angle>1e-6):
                axis =  self.r_w[0] / angle
            tmp = ti.math.sin(0.5*dt*angle)*axis
            self.r_q[0] = quaternion_multiplication(ti.Vector([ti.math.cos(0.5*dt*angle), tmp[0], tmp[1], tmp[2]]), self.r_q[0])
            quaternion_normalization(self.r_q[0])
            self.r_rotM[0] = quaternion_to_matrix(self.r_q[0])


    # update linear and angular velocity by contact impulse
    # reset rigid body position and orientation to the begining of this time step
    @ti.kernel
    def post_contact(self):
        if(self.n_contact[0]>0):
            # print("contact", self.n_contact[0])
            self.r_v[0] += self.delta_v[0]/self.n_contact[0]
            self.r_w[0] += self.delta_w[0]/self.n_contact[0]
        if(self.sleep[0]>0):
            # print('sleep contact status', self.sleep[0])
            self.r_v[0] = self.r_v_pre[0]
            self.r_w[0] = self.r_w_pre[0]


    # forward the rigid body state for one time step                
    @ti.kernel
    def forward_step(self, dt: ti.f64):
        if(self.sleep[0]<0):
            self.r_x[0] +=  dt * self.r_v[0]

            angle = ti.pow(ti.math.dot(self.r_w[0], self.r_w[0]),0.5)
            axis = ti.Vector([0.0,1.0,0.0])
            if(angle>1e-6):
                axis =  self.r_w[0] / angle
            tmp = ti.math.sin(0.5*dt*angle)*axis
            self.r_q[0] = quaternion_multiplication(ti.Vector([ti.math.cos(0.5*dt*angle), tmp[0], tmp[1], tmp[2]]), self.r_q[0])
            quaternion_normalization(self.r_q[0])
            self.r_rotM[0] = quaternion_to_matrix(self.r_q[0])

                    

    # check if the rigid body becoms sleep
    @ti.kernel
    def check_sleep(self, sleepThreshold: ti.f64): 
        bias = 0.1
        motion_bound = 2 * (ti.math.pow(ti.math.length(self.r_v[0]), 2)+ti.math.pow(ti.math.length(self.r_w[0]), 2)*ti.math.pow(self.max_distance[0],2))
        self.rwa[0] = bias * self.rwa[0] + (1-bias)*motion_bound
        if(self.rwa[0]<ti.math.pow(sleepThreshold, 2)):
            self.sleep[0] = 1.0
            self.r_v[0] = ti.Vector([0.0,0.0,0.0])
            self.r_w[0] = ti.Vector([0.0,0.0,0.0])
            print("sleep", self.delta_v[0])
            self.rwa[0] = ti.math.pow(sleepThreshold, 2)
        else:
            self.rwa[0] = motion_bound


   
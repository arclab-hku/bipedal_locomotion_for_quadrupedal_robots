import pybullet as p 
import time 
import pybullet_data



import pybullet as p
import time

# class Dog:
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version 
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally 

planeId = p.loadURDF("plane.urdf") 

p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
#p.setDefaultContactERP(0)
#urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
urdfFlags = p.URDF_USE_SELF_COLLISION
quadruped = p.loadURDF("urdf/arcdog.urdf", [0, 0, 1], [0, 0, 0, 1], flags = urdfFlags, useFixedBase=True)

#enable collision between lower legs
for j in range (p.getNumJoints(quadruped)):
        print(p.getJointInfo(quadruped,j))

lower_legs = [2,5,8,11]
for l0 in lower_legs:
    for l1 in lower_legs:
        if (l1>l0):
            enableCollision = 1
            print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
            p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

jointIds=[]
paramIds=[]

maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

for j in range (p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped,j)
    print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        jointIds.append(j)

# print(jointIds)

p.getCameraImage(480,320)
p.setRealTimeSimulation(0)

joints=[]

# while(1):
#     with open("mocap.txt","r") as filestream:
#         for line in filestream:
#             maxForce = p.readUserDebugParameter(maxForceId)
#             currentline = line.split(",")
#             frame = currentline[0]
#             t = currentline[1]
#             joints = currentline[2:8]+[0.0, 0.6, -1.2]*4

#             for j in range (18):
#                 targetPos = float(joints[j])
#                 p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)

#             p.stepSimulation()
#             for lower_leg in lower_legs:
#                 #print("points for ", quadruped, " link: ", lower_leg)
#                 pts = p.getContactPoints(quadruped,-1, lower_leg)
#                 #print("num points=",len(pts))
#                 #for pt in pts:
#                 #    print(pt[9])
#             time.sleep(1./500.)

# p.disconnect()
for j in range (p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped,j)
    js = p.getJointState(quadruped,j)
    #print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,(js[0]-0)/1))


p.setRealTimeSimulation(1)

while (1):
    for i in range(len(paramIds)):
        c = paramIds[i]
        targetPos = p.readUserDebugParameter(c)
        maxForce = p.readUserDebugParameter(maxForceId)
        p.setJointMotorControl2(quadruped,jointIds[i],p.POSITION_CONTROL,1*targetPos+0, force=maxForce)

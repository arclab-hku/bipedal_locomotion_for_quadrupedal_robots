
from urdfpy import URDF
# import os

class Agent:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.robot = URDF.load(urdf_path)

    def get_trunk(self):
        box_size = None
        for jdx, link in enumerate(self.robot.links):
            if link.name in ["trunk"]:
                box_size = self.robot.links[jdx].visuals[0].geometry.box.size
        return box_size


    def mod_leg(self,  leg_type="thigh"):
        assert (leg_type in ["thigh", "calf"])
        leg_indices = []
        if leg_type == "thigh":
            for link_idx, link in enumerate(self.robot.links):
                if link.name.split("_")[-1] == "thigh":
                    leg_indices.append(link_idx)
        elif leg_type == "calf":
            for link_idx, link in enumerate(self.robot.links):
                if link.name.split("_")[-1] == "calf":
                    leg_indices.append(link_idx)
        box_size = None
        for i in leg_indices:
            box_size = self.robot.links[i].visuals[0].geometry.box.size
        return box_size

# def main(args=None):
#     asset_files = os.listdir('urdf_set')
#     urdf_nums = 3
#     for i in range(0, urdf_nums):
#         # j = np.random.randint(0,len(asset_files))
#         morph_dog  =  Agent("urdf_set/{}".format(asset_files[i]))
#         trunk_size = morph_dog.get_trunk()
#         thigh_size = morph_dog.mod_leg(leg_type="thigh")
#         calf_size = morph_dog.mod_leg(leg_type="calf")
#
#         print(i, trunk_size, thigh_size, calf_size)
#
#
#
# if __name__ == "__main__":
#     main()
"""Inspect one two-feature angle encoding; input features are already scaled."""
import argparse
from angle_encoding import cudaq, np, MAPS, AXES, to_angles, validate, state_vector, encoded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=float,nargs=2,default=[.25,-.4])
    p.add_argument('--mapping',choices=MAPS,default='positive_half')
    p.add_argument('--axis',choices=AXES,default='ry')
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    args=p.parse_args()
    angles=to_angles(args.features,args.mapping)
    _,code=validate(angles,args.axis)
    cudaq.set_target(args.backend)
    print('scaled features:',args.features)
    print('radians:',angles)
    print(cudaq.draw(encoded,angles,code))
    print('probabilities [00,01,10,11]:',(abs(state_vector(angles,args.axis))**2).tolist())
    for name,op in [('X0',cudaq.spin.x(0)),('Y0',cudaq.spin.y(0)),('Z0',cudaq.spin.z(0))]:
        print(name,cudaq.observe(encoded,op,angles,code,shots_count=-1).expectation())

if __name__=='__main__': main()

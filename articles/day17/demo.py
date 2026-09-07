import argparse
from gradients import parameter_shift,matrix_gradient,loss_and_gradient
from articles.day14.model import cudaq,initialize


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');a=p.parse_args();cudaq.set_target(a.backend)
    w=initialize();x=[.25,-.4]
    print('df/dw parameter-shift:',parameter_shift(x,w))
    print('df/dw matrix reference:',matrix_gradient(x,w))
    loss,g=loss_and_gradient([x],[1],w)
    print('Brier loss:',loss,'dL/dw:',g)

if __name__=='__main__':main()

clear;
close;

analytical = true;

Ez_plane = permute(hdf5read('Ez_plane.h5', 'Ez_plane'), [3, 2, 1]);
Ez_plane_analytical = permute(hdf5read('Ez_plane_a.h5', 'Ez_plane_a'), [3, 2, 1]);
N = size(Ez_plane, 1);
I = size(Ez_plane, 2);
J = size(Ez_plane, 3);
    
[Y, X] = meshgrid(0:I-1, 0:J-1);

figure(1)
loops = N;

if ~analytical
    for j = 1:loops
        surf(X, Y, reshape(Ez_plane(j, :, :), [I, J]))
        axis([0 I 0 J -0.1 0.1])

        drawnow
%         pause(1)
    end
    
else
    for j = 1:loops
        ax1 = subplot(1, 2, 1);
        surf(ax1, X, Y, reshape(Ez_plane(j, :, :), [I, J]))
        axis([0 I 0 J -1 1])

        ax2 = subplot(1, 2, 2);
        surf(ax2, X, Y, reshape(Ez_plane_analytical(j, :, :), [I, J]))
        axis([0 I 0 J -1 1])

        drawnow
%         pause(1)
    end
end
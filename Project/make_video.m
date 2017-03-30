clear
close

analytical = true;  % When true: Shows analytical solution next to numerical

Ez_plane = permute(hdf5read('Ez_plane.h5', 'Ez_plane'), [3, 2, 1]);
Ez_plane_analytical = permute(hdf5read('Ez_plane_a.h5', 'Ez_plane_a'), [3, 2, 1]);
N = size(Ez_plane, 1);
M = size(Ez_plane, 2);
    
[Y, X] = meshgrid(0:M-1, 0:M-1);

figure(1)
set_latex_interpreter()

loops = N;

v = VideoWriter('movies/wave_movie.avi');
open(v)

if ~analytical
    z_ax_max = max(max(Ez_plane(:)));
    z_ax_min = min(min(Ez_plane(:)));
    for j = 1:loops
        surf(X, Y, reshape(Ez_plane(j, :, :), [M, M]))
        axis([0 M 0 M z_ax_min z_ax_max])
        
        % Labels and title
        if j == 1
%             title('z-component of electrical field in plane at constant $z$', 'HandleVisibility', 'off')
            set_labs_and_title('Numerical', '$E_z$')
        end
        set( gca , 'NextPlot' , 'replacechildren' ) 

        writeVideo(v ,getframe(gcf))
        drawnow
    end
    
else
    z_ax_max = max(max(Ez_plane(:)), max(Ez_plane_analytical(:)));
    z_ax_min = min(min(Ez_plane(:)), min(Ez_plane_analytical(:)));
    for j = 1:loops
        ax1 = subplot(1, 2, 1);
        surf(ax1, X, Y, reshape(Ez_plane(j, :, :), [M, M]))
        axis([0 M 0 M z_ax_min z_ax_max])
        
        % Labels and subtitle
        if j == 1
            set_labs_and_title('Numerical', '$E_z$')
        end
        set( gca , 'NextPlot' , 'replacechildren' )

        ax2 = subplot(1, 2, 2);
        surf(ax2, X, Y, reshape(Ez_plane_analytical(j, :, :), [M, M]))
        axis([0 M 0 M z_ax_min z_ax_max])
       
        % Labels and subtitle
        if j == 1
            set_labs_and_title('Analytical', '$E_z$')
        end
        set( gca , 'NextPlot' , 'replacechildren' ) ;
        
        % Title
%         set(gcf, 'NextPlot', 'add')
%         axes;
%         t = title('z-component of electrical field in plane at constant $z$');
%         set(gca, 'Visible', 'off')
%         set(t, 'Visible', 'on')
        
        writeVideo(v ,getframe(gcf))
        drawnow
    end
end

close(v)
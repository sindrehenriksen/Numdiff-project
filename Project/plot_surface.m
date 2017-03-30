clear
close

analytical = true;  % When true: Shows analytical solution next to numerical
plot_slices = false;  % Plots field in intersecting planes at different times

Ez_plane = permute(hdf5read('Ez_plane.h5', 'Ez_plane'), [2, 1, 3]);
Ez_plane_analytical = permute(hdf5read('Ez_plane_a.h5', 'Ez_plane_a'), [2, 1, 3]);
M = size(Ez_plane, 1);
N = size(Ez_plane, 3);
    
[Y, X] = meshgrid(0:M-1, 0:M-1);

fig = figure(1);
set_latex_interpreter()
planes = 5;
t = int8(size(Ez_plane, 3) * 1 / 20);

if ~analytical
    if plot_slices
        slice(Ez_plane, [], [], floor(linspace(1, N, planes)))
%         c = colorbar;
%         ylabel(c, '$E_z$')
        shading  flat
        set_labs_and_title('Numerical', '$t$')
        
    else
        z_ax_max = max(max(Ez_plane(:)));
        z_ax_min = min(min(Ez_plane(:)));
        surf(X, Y, reshape(Ez_plane(:, :, t), [M, M]))
        axis([0 M 0 M z_ax_min z_ax_max])
        set_labs_and_title('Numerical', '$E_z$')
    end
    
else
    if plot_slices
        ax1 = subplot(1, 2, 1);
        slice(ax1, Ez_plane, [], [], floor(linspace(1, N, planes)))
        shading  flat
        set_labs_and_title('Numerical', '$t$')

        ax1 = subplot(1, 2, 2);
        slice(ax1, Ez_plane_analytical, [], [], floor(linspace(1, N, planes)))
        shading  flat
        set_labs_and_title('Analytical', '$t$')
        
    else
        z_ax_max = max(max(Ez_plane(:)), max(Ez_plane_analytical(:)));
        z_ax_min = min(min(Ez_plane(:)), min(Ez_plane_analytical(:)));
        
        ax1 = subplot(1, 2, 1);
        surf(ax1, X, Y, reshape(Ez_plane(:, :, t), [M, M]))
        axis([0 M 0 M z_ax_min z_ax_max])
        set_labs_and_title('Numerical', '$E_z$')
        
        ax2 = subplot(1, 2, 2);
        surf(ax2, X, Y, reshape(Ez_plane_analytical(:, :, t), [M, M]))
        axis([0 M 0 M z_ax_min z_ax_max])
        set_labs_and_title('Analytical', '$E_z$')
    end
end

saveas(fig, 'figures/wave', 'epsc')
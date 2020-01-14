clear;

% x = uint16(randi([0,255],[256,256]));
% 
% figure(1); clf;
% imagesc(x);
% axis off
% 
% return


h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'testAnimated.gif';
for n = 1:20
    x = uint16(randi([0,255],[256,256]));

    imagesc(x);
    axis off

    
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
end
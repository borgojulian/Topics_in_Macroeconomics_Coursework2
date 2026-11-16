function y = markovchain(Trans,T,s0,me)

%Syntax:	y=markovchain(Trans,T,s0,me);
%
%This function generates a simulation from a Markov chain.
%Trans is the transition matrix, T is the number of periods 
%to be simulated, s0 is the initial state, with one as default,
%me is the method (two possibilities are included, with the first
%as default), y is the shock realization.
%
%Eva Carceles-Poveda 2003

%Checking the mistakes from the inputs
[s1 s2]=size(Trans);

if nargin == 1;
  s0=1;
  T=100;
  me=1;
end
if nargin == 2;
   s0=1;
   me=1;
end
if nargin == 3;
   me=1;
end

%check that Trans and s0 are well defined
if s1 ~= s2;
  disp('Transition matrix must be square');
  return
end

for k=1:s1;
  if sum(Trans(k,:)) ~= 1;
    disp(['Row ',num2str(k),' does not sum to one']);
    disp(['Normalizing row ',num2str(k),'']);
    Trans(k,:)=Trans(k,:)/sum(Trans(k,:));
  end
end

if s0 < 1 |s0 > s1;
  disp(['Initial state ',num2str(s0),' is out of range']);
  disp(['Initial state defaulting to 1']);
  s0=1;
end

%Creating the shock realizations

X=rand(T,1);
y(1,1)=s0;
if me==1%use method 1
   for i=2:T
      for j=1:s1   
         if X(i-1)<sum(Trans(y(i-1),1:j)) 
            break
      		end
        j=j+1;
      end
      y(i,1)=j;
   end
elseif me==2
  	s=zeros(s1,1);
	s(s0)=1;
	cum=Trans*triu(ones(size(Trans)));
	for k=1:length(X)
  		state(:,k)=s;
  		ppi=[0 s'*cum];
  		s=((X(k)<=ppi(2:s1+1)).*(X(k)>ppi(1:s1)))';
	end
   y=[1:s1]*state;
end

 
??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.1-0-gfcc4b966f18??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:0*
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
:0*
dtype0
?
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H*!
shared_nameconv2d_29/kernel
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*&
_output_shapes
:0H*
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:H*
dtype0
?
conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H?*!
shared_nameconv2d_30/kernel
~
$conv2d_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_30/kernel*'
_output_shapes
:H?*
dtype0
u
conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_30/bias
n
"conv2d_30/bias/Read/ReadVariableOpReadVariableOpconv2d_30/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_8/gamma
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_8/moving_mean
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_8/moving_variance
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_31/kernel

$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_31/bias
n
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes	
:?*
dtype0
?
conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?H*!
shared_nameconv2d_32/kernel
~
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*'
_output_shapes
:?H*
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
:H*
dtype0
?
conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H0*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
:H0*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:0*
dtype0
?
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_9/gamma
?
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:0*
dtype0
?
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_9/beta
?
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:0*
dtype0
?
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_9/moving_mean
?
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:0*
dtype0
?
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_9/moving_variance
?
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:0*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameAdam/conv2d_28/kernel/m
?
+Adam/conv2d_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/m*&
_output_shapes
:0*
dtype0
?
Adam/conv2d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameAdam/conv2d_28/bias/m
{
)Adam/conv2d_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/m*
_output_shapes
:0*
dtype0
?
Adam/conv2d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H*(
shared_nameAdam/conv2d_29/kernel/m
?
+Adam/conv2d_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/m*&
_output_shapes
:0H*
dtype0
?
Adam/conv2d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*&
shared_nameAdam/conv2d_29/bias/m
{
)Adam/conv2d_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/m*
_output_shapes
:H*
dtype0
?
Adam/conv2d_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H?*(
shared_nameAdam/conv2d_30/kernel/m
?
+Adam/conv2d_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/kernel/m*'
_output_shapes
:H?*
dtype0
?
Adam/conv2d_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_30/bias/m
|
)Adam/conv2d_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_8/gamma/m
?
6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/m*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_8/beta/m
?
5Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_31/kernel/m
?
+Adam/conv2d_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_31/bias/m
|
)Adam/conv2d_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?H*(
shared_nameAdam/conv2d_32/kernel/m
?
+Adam/conv2d_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/m*'
_output_shapes
:?H*
dtype0
?
Adam/conv2d_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*&
shared_nameAdam/conv2d_32/bias/m
{
)Adam/conv2d_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/m*
_output_shapes
:H*
dtype0
?
Adam/conv2d_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H0*(
shared_nameAdam/conv2d_33/kernel/m
?
+Adam/conv2d_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/m*&
_output_shapes
:H0*
dtype0
?
Adam/conv2d_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameAdam/conv2d_33/bias/m
{
)Adam/conv2d_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/m*
_output_shapes
:0*
dtype0
?
"Adam/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/batch_normalization_9/gamma/m
?
6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/m*
_output_shapes
:0*
dtype0
?
!Adam/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/batch_normalization_9/beta/m
?
5Adam/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/m*
_output_shapes
:0*
dtype0
?
Adam/conv2d_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameAdam/conv2d_34/kernel/m
?
+Adam/conv2d_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/m*&
_output_shapes
:0*
dtype0
?
Adam/conv2d_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_34/bias/m
{
)Adam/conv2d_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameAdam/conv2d_28/kernel/v
?
+Adam/conv2d_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/v*&
_output_shapes
:0*
dtype0
?
Adam/conv2d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameAdam/conv2d_28/bias/v
{
)Adam/conv2d_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/v*
_output_shapes
:0*
dtype0
?
Adam/conv2d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H*(
shared_nameAdam/conv2d_29/kernel/v
?
+Adam/conv2d_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/v*&
_output_shapes
:0H*
dtype0
?
Adam/conv2d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*&
shared_nameAdam/conv2d_29/bias/v
{
)Adam/conv2d_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/v*
_output_shapes
:H*
dtype0
?
Adam/conv2d_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H?*(
shared_nameAdam/conv2d_30/kernel/v
?
+Adam/conv2d_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/kernel/v*'
_output_shapes
:H?*
dtype0
?
Adam/conv2d_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_30/bias/v
|
)Adam/conv2d_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_8/gamma/v
?
6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/v*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_8/beta/v
?
5Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_31/kernel/v
?
+Adam/conv2d_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_31/bias/v
|
)Adam/conv2d_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?H*(
shared_nameAdam/conv2d_32/kernel/v
?
+Adam/conv2d_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/v*'
_output_shapes
:?H*
dtype0
?
Adam/conv2d_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*&
shared_nameAdam/conv2d_32/bias/v
{
)Adam/conv2d_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/v*
_output_shapes
:H*
dtype0
?
Adam/conv2d_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H0*(
shared_nameAdam/conv2d_33/kernel/v
?
+Adam/conv2d_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/v*&
_output_shapes
:H0*
dtype0
?
Adam/conv2d_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameAdam/conv2d_33/bias/v
{
)Adam/conv2d_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/v*
_output_shapes
:0*
dtype0
?
"Adam/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/batch_normalization_9/gamma/v
?
6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/v*
_output_shapes
:0*
dtype0
?
!Adam/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/batch_normalization_9/beta/v
?
5Adam/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/v*
_output_shapes
:0*
dtype0
?
Adam/conv2d_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameAdam/conv2d_34/kernel/v
?
+Adam/conv2d_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/v*&
_output_shapes
:0*
dtype0
?
Adam/conv2d_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_34/bias/v
{
)Adam/conv2d_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?j
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?j
value?jB?j B?j
?
encoder
decoder
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer-5
regularization_losses
	variables
trainable_variables
	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
regularization_losses
	variables
trainable_variables
	keras_api
?
iter

beta_1

beta_2
	 decay
!learning_rate"m?#m?$m?%m?&m?'m?(m?)m?,m?-m?.m?/m?0m?1m?2m?3m?6m?7m?"v?#v?$v?%v?&v?'v?(v?)v?,v?-v?.v?/v?0v?1v?2v?3v?6v?7v?
 
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
?
"0
#1
$2
%3
&4
'5
(6
)7
,8
-9
.10
/11
012
113
214
315
616
717
?
regularization_losses
8layer_regularization_losses

9layers
:non_trainable_variables
;metrics
	variables
trainable_variables
<layer_metrics
 
h

"kernel
#bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
h

$kernel
%bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
h

&kernel
'bias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
?
Iaxis
	(gamma
)beta
*moving_mean
+moving_variance
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
R
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
 
F
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
8
"0
#1
$2
%3
&4
'5
(6
)7
?
regularization_losses
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
Ymetrics
	variables
trainable_variables
Zlayer_metrics
?
[_inbound_nodes

,kernel
-bias
\_outbound_nodes
]regularization_losses
^	variables
_trainable_variables
`	keras_api
?
a_inbound_nodes

.kernel
/bias
b_outbound_nodes
cregularization_losses
d	variables
etrainable_variables
f	keras_api
?
g_inbound_nodes

0kernel
1bias
h_outbound_nodes
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
?
m_inbound_nodes
naxis
	2gamma
3beta
4moving_mean
5moving_variance
o_outbound_nodes
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
{
t_inbound_nodes
u_outbound_nodes
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
|
z_inbound_nodes

6kernel
7bias
{regularization_losses
|	variables
}trainable_variables
~	keras_api
 
V
,0
-1
.2
/3
04
15
26
37
48
59
610
711
F
,0
-1
.2
/3
04
15
26
37
68
79
?
regularization_losses
layer_regularization_losses
?layers
?non_trainable_variables
?metrics
	variables
trainable_variables
?layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_28/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_28/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_29/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_29/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_30/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_30/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_8/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_8/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_8/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_8/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_31/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_31/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_32/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_32/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_33/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_33/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_9/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_9/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_9/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_9/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_34/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_34/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

*0
+1
42
53

?0
?1
 
 

"0
#1

"0
#1
?
=regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
>	variables
?trainable_variables
?layer_metrics
 

$0
%1

$0
%1
?
Aregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
B	variables
Ctrainable_variables
?layer_metrics
 

&0
'1

&0
'1
?
Eregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
F	variables
Gtrainable_variables
?layer_metrics
 
 

(0
)1
*2
+3

(0
)1
?
Jregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
K	variables
Ltrainable_variables
?layer_metrics
 
 
 
?
Nregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
O	variables
Ptrainable_variables
?layer_metrics
 
 
 
?
Rregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
S	variables
Ttrainable_variables
?layer_metrics
 
*
	0

1
2
3
4
5

*0
+1
 
 
 
 
 

,0
-1

,0
-1
?
]regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
^	variables
_trainable_variables
?layer_metrics
 
 
 

.0
/1

.0
/1
?
cregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
d	variables
etrainable_variables
?layer_metrics
 
 
 

00
11

00
11
?
iregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
j	variables
ktrainable_variables
?layer_metrics
 
 
 
 

20
31
42
53

20
31
?
pregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
q	variables
rtrainable_variables
?layer_metrics
 
 
 
 
 
?
vregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
w	variables
xtrainable_variables
?layer_metrics
 
 

60
71

60
71
?
{regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
|	variables
}trainable_variables
?layer_metrics
 
*
0
1
2
3
4
5

40
51
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

*0
+1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

40
51
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
om
VARIABLE_VALUEAdam/conv2d_28/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_28/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_29/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_29/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_30/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_30/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_8/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_31/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_31/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_32/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_32/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_33/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_33/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_9/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_34/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_34/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_28/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_28/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_29/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_29/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_30/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_30/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_8/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_31/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_31/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_32/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_32/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_33/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_33/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_9/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2d_34/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_34/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_34/kernelconv2d_34/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_34068
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp$conv2d_30/kernel/Read/ReadVariableOp"conv2d_30/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_28/kernel/m/Read/ReadVariableOp)Adam/conv2d_28/bias/m/Read/ReadVariableOp+Adam/conv2d_29/kernel/m/Read/ReadVariableOp)Adam/conv2d_29/bias/m/Read/ReadVariableOp+Adam/conv2d_30/kernel/m/Read/ReadVariableOp)Adam/conv2d_30/bias/m/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_8/beta/m/Read/ReadVariableOp+Adam/conv2d_31/kernel/m/Read/ReadVariableOp)Adam/conv2d_31/bias/m/Read/ReadVariableOp+Adam/conv2d_32/kernel/m/Read/ReadVariableOp)Adam/conv2d_32/bias/m/Read/ReadVariableOp+Adam/conv2d_33/kernel/m/Read/ReadVariableOp)Adam/conv2d_33/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp+Adam/conv2d_34/kernel/m/Read/ReadVariableOp)Adam/conv2d_34/bias/m/Read/ReadVariableOp+Adam/conv2d_28/kernel/v/Read/ReadVariableOp)Adam/conv2d_28/bias/v/Read/ReadVariableOp+Adam/conv2d_29/kernel/v/Read/ReadVariableOp)Adam/conv2d_29/bias/v/Read/ReadVariableOp+Adam/conv2d_30/kernel/v/Read/ReadVariableOp)Adam/conv2d_30/bias/v/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_8/beta/v/Read/ReadVariableOp+Adam/conv2d_31/kernel/v/Read/ReadVariableOp)Adam/conv2d_31/bias/v/Read/ReadVariableOp+Adam/conv2d_32/kernel/v/Read/ReadVariableOp)Adam/conv2d_32/bias/v/Read/ReadVariableOp+Adam/conv2d_33/kernel/v/Read/ReadVariableOp)Adam/conv2d_33/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp+Adam/conv2d_34/kernel/v/Read/ReadVariableOp)Adam/conv2d_34/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_35774
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_34/kernelconv2d_34/biastotalcounttotal_1count_1Adam/conv2d_28/kernel/mAdam/conv2d_28/bias/mAdam/conv2d_29/kernel/mAdam/conv2d_29/bias/mAdam/conv2d_30/kernel/mAdam/conv2d_30/bias/m"Adam/batch_normalization_8/gamma/m!Adam/batch_normalization_8/beta/mAdam/conv2d_31/kernel/mAdam/conv2d_31/bias/mAdam/conv2d_32/kernel/mAdam/conv2d_32/bias/mAdam/conv2d_33/kernel/mAdam/conv2d_33/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/conv2d_34/kernel/mAdam/conv2d_34/bias/mAdam/conv2d_28/kernel/vAdam/conv2d_28/bias/vAdam/conv2d_29/kernel/vAdam/conv2d_29/bias/vAdam/conv2d_30/kernel/vAdam/conv2d_30/bias/v"Adam/batch_normalization_8/gamma/v!Adam/batch_normalization_8/beta/vAdam/conv2d_31/kernel/vAdam/conv2d_31/bias/vAdam/conv2d_32/kernel/vAdam/conv2d_32/bias/vAdam/conv2d_33/kernel/vAdam/conv2d_33/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/conv2d_34/kernel/vAdam/conv2d_34/bias/v*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_35985??
?
?
5__inference_batch_normalization_9_layer_call_fn_35466

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_332732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_34068
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_327332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35504

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????0:::::Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_35138

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_31_layer_call_and_return_conditional_losses_33318

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????:::Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?v
?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34261
input_19
5sequential_8_conv2d_28_conv2d_readvariableop_resource:
6sequential_8_conv2d_28_biasadd_readvariableop_resource9
5sequential_8_conv2d_29_conv2d_readvariableop_resource:
6sequential_8_conv2d_29_biasadd_readvariableop_resource9
5sequential_8_conv2d_30_conv2d_readvariableop_resource:
6sequential_8_conv2d_30_biasadd_readvariableop_resource>
:sequential_8_batch_normalization_8_readvariableop_resource@
<sequential_8_batch_normalization_8_readvariableop_1_resourceO
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceQ
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource9
5sequential_9_conv2d_31_conv2d_readvariableop_resource:
6sequential_9_conv2d_31_biasadd_readvariableop_resource9
5sequential_9_conv2d_32_conv2d_readvariableop_resource:
6sequential_9_conv2d_32_biasadd_readvariableop_resource9
5sequential_9_conv2d_33_conv2d_readvariableop_resource:
6sequential_9_conv2d_33_biasadd_readvariableop_resource>
:sequential_9_batch_normalization_9_readvariableop_resource@
<sequential_9_batch_normalization_9_readvariableop_1_resourceO
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceQ
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource9
5sequential_9_conv2d_34_conv2d_readvariableop_resource:
6sequential_9_conv2d_34_biasadd_readvariableop_resource
identity??
,sequential_8/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,sequential_8/conv2d_28/Conv2D/ReadVariableOp?
sequential_8/conv2d_28/Conv2DConv2Dinput_14sequential_8/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
sequential_8/conv2d_28/Conv2D?
-sequential_8/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_8/conv2d_28/BiasAdd/ReadVariableOp?
sequential_8/conv2d_28/BiasAddBiasAdd&sequential_8/conv2d_28/Conv2D:output:05sequential_8/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02 
sequential_8/conv2d_28/BiasAdd?
sequential_8/conv2d_28/ReluRelu'sequential_8/conv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
sequential_8/conv2d_28/Relu?
,sequential_8/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02.
,sequential_8/conv2d_29/Conv2D/ReadVariableOp?
sequential_8/conv2d_29/Conv2DConv2D)sequential_8/conv2d_28/Relu:activations:04sequential_8/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
sequential_8/conv2d_29/Conv2D?
-sequential_8/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_8/conv2d_29/BiasAdd/ReadVariableOp?
sequential_8/conv2d_29/BiasAddBiasAdd&sequential_8/conv2d_29/Conv2D:output:05sequential_8/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2 
sequential_8/conv2d_29/BiasAdd?
sequential_8/conv2d_29/ReluRelu'sequential_8/conv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
sequential_8/conv2d_29/Relu?
,sequential_8/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_30_conv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02.
,sequential_8/conv2d_30/Conv2D/ReadVariableOp?
sequential_8/conv2d_30/Conv2DConv2D)sequential_8/conv2d_29/Relu:activations:04sequential_8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential_8/conv2d_30/Conv2D?
-sequential_8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_30/BiasAdd/ReadVariableOp?
sequential_8/conv2d_30/BiasAddBiasAdd&sequential_8/conv2d_30/Conv2D:output:05sequential_8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2 
sequential_8/conv2d_30/BiasAdd?
sequential_8/conv2d_30/ReluRelu'sequential_8/conv2d_30/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential_8/conv2d_30/Relu?
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp?
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1?
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3)sequential_8/conv2d_30/Relu:activations:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3?
$sequential_8/max_pooling2d_4/MaxPoolMaxPool7sequential_8/batch_normalization_8/FusedBatchNormV3:y:0*2
_output_shapes 
:????????????*
ksize
*
paddingSAME*
strides
2&
$sequential_8/max_pooling2d_4/MaxPool?
sequential_8/dropout_4/IdentityIdentity-sequential_8/max_pooling2d_4/MaxPool:output:0*
T0*2
_output_shapes 
:????????????2!
sequential_8/dropout_4/Identity?
,sequential_9/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,sequential_9/conv2d_31/Conv2D/ReadVariableOp?
sequential_9/conv2d_31/Conv2DConv2D(sequential_8/dropout_4/Identity:output:04sequential_9/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential_9/conv2d_31/Conv2D?
-sequential_9/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_9/conv2d_31/BiasAdd/ReadVariableOp?
sequential_9/conv2d_31/BiasAddBiasAdd&sequential_9/conv2d_31/Conv2D:output:05sequential_9/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2 
sequential_9/conv2d_31/BiasAdd?
sequential_9/conv2d_31/ReluRelu'sequential_9/conv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential_9/conv2d_31/Relu?
,sequential_9/conv2d_32/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02.
,sequential_9/conv2d_32/Conv2D/ReadVariableOp?
sequential_9/conv2d_32/Conv2DConv2D)sequential_9/conv2d_31/Relu:activations:04sequential_9/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
sequential_9/conv2d_32/Conv2D?
-sequential_9/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_9/conv2d_32/BiasAdd/ReadVariableOp?
sequential_9/conv2d_32/BiasAddBiasAdd&sequential_9/conv2d_32/Conv2D:output:05sequential_9/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2 
sequential_9/conv2d_32/BiasAdd?
sequential_9/conv2d_32/ReluRelu'sequential_9/conv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
sequential_9/conv2d_32/Relu?
,sequential_9/conv2d_33/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02.
,sequential_9/conv2d_33/Conv2D/ReadVariableOp?
sequential_9/conv2d_33/Conv2DConv2D)sequential_9/conv2d_32/Relu:activations:04sequential_9/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
sequential_9/conv2d_33/Conv2D?
-sequential_9/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_9/conv2d_33/BiasAdd/ReadVariableOp?
sequential_9/conv2d_33/BiasAddBiasAdd&sequential_9/conv2d_33/Conv2D:output:05sequential_9/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02 
sequential_9/conv2d_33/BiasAdd?
sequential_9/conv2d_33/ReluRelu'sequential_9/conv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
sequential_9/conv2d_33/Relu?
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp?
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1?
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3)sequential_9/conv2d_33/Relu:activations:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3?
"sequential_9/up_sampling2d_4/ShapeShape7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2$
"sequential_9/up_sampling2d_4/Shape?
0sequential_9/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_9/up_sampling2d_4/strided_slice/stack?
2sequential_9/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_9/up_sampling2d_4/strided_slice/stack_1?
2sequential_9/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_9/up_sampling2d_4/strided_slice/stack_2?
*sequential_9/up_sampling2d_4/strided_sliceStridedSlice+sequential_9/up_sampling2d_4/Shape:output:09sequential_9/up_sampling2d_4/strided_slice/stack:output:0;sequential_9/up_sampling2d_4/strided_slice/stack_1:output:0;sequential_9/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_9/up_sampling2d_4/strided_slice?
"sequential_9/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_9/up_sampling2d_4/Const?
 sequential_9/up_sampling2d_4/mulMul3sequential_9/up_sampling2d_4/strided_slice:output:0+sequential_9/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2"
 sequential_9/up_sampling2d_4/mul?
9sequential_9/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0$sequential_9/up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2;
9sequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor?
,sequential_9/conv2d_34/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,sequential_9/conv2d_34/Conv2D/ReadVariableOp?
sequential_9/conv2d_34/Conv2DConv2DJsequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:04sequential_9/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
sequential_9/conv2d_34/Conv2D?
-sequential_9/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_9/conv2d_34/BiasAdd/ReadVariableOp?
sequential_9/conv2d_34/BiasAddBiasAdd&sequential_9/conv2d_34/Conv2D:output:05sequential_9/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
sequential_9/conv2d_34/BiasAdd?
sequential_9/conv2d_34/SigmoidSigmoid'sequential_9/conv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2 
sequential_9/conv2d_34/Sigmoid?
IdentityIdentity"sequential_9/conv2d_34/Sigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????:::::::::::::::::::::::Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35225

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3s
IdentityIdentityFusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????:::::Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?C
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_35015
conv2d_31_input,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource,
(conv2d_32_conv2d_readvariableop_resource-
)conv2d_32_biasadd_readvariableop_resource,
(conv2d_33_conv2d_readvariableop_resource-
)conv2d_33_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource
identity??$batch_normalization_9/AssignNewValue?&batch_normalization_9/AssignNewValue_1?
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_31/Conv2D/ReadVariableOp?
conv2d_31/Conv2DConv2Dconv2d_31_input'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_31/Conv2D?
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp?
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_31/BiasAdd?
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
conv2d_31/Relu?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02!
conv2d_32/Conv2D/ReadVariableOp?
conv2d_32/Conv2DConv2Dconv2d_31/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
conv2d_32/Conv2D?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2
conv2d_32/BiasAdd?
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
conv2d_32/Relu?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02!
conv2d_33/Conv2D/ReadVariableOp?
conv2d_33/Conv2DConv2Dconv2d_32/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_33/Conv2D?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_33/BiasAdd?
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_33/Relu?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_33/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_9/FusedBatchNormV3?
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue?
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1?
up_sampling2d_4/ShapeShape*batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape?
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack?
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1?
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2?
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor*batch_normalization_9/FusedBatchNormV3:y:0up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_34/BiasAdd?
conv2d_34/SigmoidSigmoidconv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_34/Sigmoid?
IdentityIdentityconv2d_34/Sigmoid:y:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????::::::::::::2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:c _
2
_output_shapes 
:????????????
)
_user_specified_nameconv2d_31_input
?:
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_35069
conv2d_31_input,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource,
(conv2d_32_conv2d_readvariableop_resource-
)conv2d_32_biasadd_readvariableop_resource,
(conv2d_33_conv2d_readvariableop_resource-
)conv2d_33_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource
identity??
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_31/Conv2D/ReadVariableOp?
conv2d_31/Conv2DConv2Dconv2d_31_input'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_31/Conv2D?
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp?
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_31/BiasAdd?
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
conv2d_31/Relu?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02!
conv2d_32/Conv2D/ReadVariableOp?
conv2d_32/Conv2DConv2Dconv2d_31/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
conv2d_32/Conv2D?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2
conv2d_32/BiasAdd?
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
conv2d_32/Relu?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02!
conv2d_33/Conv2D/ReadVariableOp?
conv2d_33/Conv2DConv2Dconv2d_32/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_33/Conv2D?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_33/BiasAdd?
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_33/Relu?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_33/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3?
up_sampling2d_4/ShapeShape*batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape?
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack?
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1?
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2?
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor*batch_normalization_9/FusedBatchNormV3:y:0up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_34/BiasAdd?
conv2d_34/SigmoidSigmoidconv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_34/Sigmoids
IdentityIdentityconv2d_34/Sigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????:::::::::::::c _
2
_output_shapes 
:????????????
)
_user_specified_nameconv2d_31_input
?
?
5__inference_batch_normalization_8_layer_call_fn_35251

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_329712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_4_layer_call_fn_35342

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_330252
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_33_layer_call_and_return_conditional_losses_33372

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????H:::Y U
1
_output_shapes
:???????????H
 
_user_specified_nameinputs
?	
?
,__inference_sequential_8_layer_call_fn_33125
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_331022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35271

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_8_layer_call_fn_35302

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_327952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_denoising_autoencoder_4_layer_call_fn_34359
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_339132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
~
)__inference_conv2d_32_layer_call_fn_35382

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_333452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????H2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32795

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_denoising_autoencoder_4_layer_call_fn_34601
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_339132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35422

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
??
?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34461
x9
5sequential_8_conv2d_28_conv2d_readvariableop_resource:
6sequential_8_conv2d_28_biasadd_readvariableop_resource9
5sequential_8_conv2d_29_conv2d_readvariableop_resource:
6sequential_8_conv2d_29_biasadd_readvariableop_resource9
5sequential_8_conv2d_30_conv2d_readvariableop_resource:
6sequential_8_conv2d_30_biasadd_readvariableop_resource>
:sequential_8_batch_normalization_8_readvariableop_resource@
<sequential_8_batch_normalization_8_readvariableop_1_resourceO
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceQ
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource9
5sequential_9_conv2d_31_conv2d_readvariableop_resource:
6sequential_9_conv2d_31_biasadd_readvariableop_resource9
5sequential_9_conv2d_32_conv2d_readvariableop_resource:
6sequential_9_conv2d_32_biasadd_readvariableop_resource9
5sequential_9_conv2d_33_conv2d_readvariableop_resource:
6sequential_9_conv2d_33_biasadd_readvariableop_resource>
:sequential_9_batch_normalization_9_readvariableop_resource@
<sequential_9_batch_normalization_9_readvariableop_1_resourceO
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceQ
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource9
5sequential_9_conv2d_34_conv2d_readvariableop_resource:
6sequential_9_conv2d_34_biasadd_readvariableop_resource
identity??1sequential_8/batch_normalization_8/AssignNewValue?3sequential_8/batch_normalization_8/AssignNewValue_1?1sequential_9/batch_normalization_9/AssignNewValue?3sequential_9/batch_normalization_9/AssignNewValue_1?
,sequential_8/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,sequential_8/conv2d_28/Conv2D/ReadVariableOp?
sequential_8/conv2d_28/Conv2DConv2Dx4sequential_8/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
sequential_8/conv2d_28/Conv2D?
-sequential_8/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_8/conv2d_28/BiasAdd/ReadVariableOp?
sequential_8/conv2d_28/BiasAddBiasAdd&sequential_8/conv2d_28/Conv2D:output:05sequential_8/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02 
sequential_8/conv2d_28/BiasAdd?
sequential_8/conv2d_28/ReluRelu'sequential_8/conv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
sequential_8/conv2d_28/Relu?
,sequential_8/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02.
,sequential_8/conv2d_29/Conv2D/ReadVariableOp?
sequential_8/conv2d_29/Conv2DConv2D)sequential_8/conv2d_28/Relu:activations:04sequential_8/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
sequential_8/conv2d_29/Conv2D?
-sequential_8/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_8/conv2d_29/BiasAdd/ReadVariableOp?
sequential_8/conv2d_29/BiasAddBiasAdd&sequential_8/conv2d_29/Conv2D:output:05sequential_8/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2 
sequential_8/conv2d_29/BiasAdd?
sequential_8/conv2d_29/ReluRelu'sequential_8/conv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
sequential_8/conv2d_29/Relu?
,sequential_8/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_30_conv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02.
,sequential_8/conv2d_30/Conv2D/ReadVariableOp?
sequential_8/conv2d_30/Conv2DConv2D)sequential_8/conv2d_29/Relu:activations:04sequential_8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential_8/conv2d_30/Conv2D?
-sequential_8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_30/BiasAdd/ReadVariableOp?
sequential_8/conv2d_30/BiasAddBiasAdd&sequential_8/conv2d_30/Conv2D:output:05sequential_8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2 
sequential_8/conv2d_30/BiasAdd?
sequential_8/conv2d_30/ReluRelu'sequential_8/conv2d_30/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential_8/conv2d_30/Relu?
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp?
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1?
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3)sequential_8/conv2d_30/Relu:activations:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<25
3sequential_8/batch_normalization_8/FusedBatchNormV3?
1sequential_8/batch_normalization_8/AssignNewValueAssignVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource@sequential_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_8/batch_normalization_8/AssignNewValue?
3sequential_8/batch_normalization_8/AssignNewValue_1AssignVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceDsequential_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0E^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_8/batch_normalization_8/AssignNewValue_1?
$sequential_8/max_pooling2d_4/MaxPoolMaxPool7sequential_8/batch_normalization_8/FusedBatchNormV3:y:0*2
_output_shapes 
:????????????*
ksize
*
paddingSAME*
strides
2&
$sequential_8/max_pooling2d_4/MaxPool?
$sequential_8/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_8/dropout_4/dropout/Const?
"sequential_8/dropout_4/dropout/MulMul-sequential_8/max_pooling2d_4/MaxPool:output:0-sequential_8/dropout_4/dropout/Const:output:0*
T0*2
_output_shapes 
:????????????2$
"sequential_8/dropout_4/dropout/Mul?
$sequential_8/dropout_4/dropout/ShapeShape-sequential_8/max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_8/dropout_4/dropout/Shape?
;sequential_8/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_8/dropout_4/dropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype02=
;sequential_8/dropout_4/dropout/random_uniform/RandomUniform?
-sequential_8/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_8/dropout_4/dropout/GreaterEqual/y?
+sequential_8/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_8/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_8/dropout_4/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2-
+sequential_8/dropout_4/dropout/GreaterEqual?
#sequential_8/dropout_4/dropout/CastCast/sequential_8/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2%
#sequential_8/dropout_4/dropout/Cast?
$sequential_8/dropout_4/dropout/Mul_1Mul&sequential_8/dropout_4/dropout/Mul:z:0'sequential_8/dropout_4/dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2&
$sequential_8/dropout_4/dropout/Mul_1?
,sequential_9/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,sequential_9/conv2d_31/Conv2D/ReadVariableOp?
sequential_9/conv2d_31/Conv2DConv2D(sequential_8/dropout_4/dropout/Mul_1:z:04sequential_9/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential_9/conv2d_31/Conv2D?
-sequential_9/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_9/conv2d_31/BiasAdd/ReadVariableOp?
sequential_9/conv2d_31/BiasAddBiasAdd&sequential_9/conv2d_31/Conv2D:output:05sequential_9/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2 
sequential_9/conv2d_31/BiasAdd?
sequential_9/conv2d_31/ReluRelu'sequential_9/conv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential_9/conv2d_31/Relu?
,sequential_9/conv2d_32/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02.
,sequential_9/conv2d_32/Conv2D/ReadVariableOp?
sequential_9/conv2d_32/Conv2DConv2D)sequential_9/conv2d_31/Relu:activations:04sequential_9/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
sequential_9/conv2d_32/Conv2D?
-sequential_9/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_9/conv2d_32/BiasAdd/ReadVariableOp?
sequential_9/conv2d_32/BiasAddBiasAdd&sequential_9/conv2d_32/Conv2D:output:05sequential_9/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2 
sequential_9/conv2d_32/BiasAdd?
sequential_9/conv2d_32/ReluRelu'sequential_9/conv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
sequential_9/conv2d_32/Relu?
,sequential_9/conv2d_33/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02.
,sequential_9/conv2d_33/Conv2D/ReadVariableOp?
sequential_9/conv2d_33/Conv2DConv2D)sequential_9/conv2d_32/Relu:activations:04sequential_9/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
sequential_9/conv2d_33/Conv2D?
-sequential_9/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_9/conv2d_33/BiasAdd/ReadVariableOp?
sequential_9/conv2d_33/BiasAddBiasAdd&sequential_9/conv2d_33/Conv2D:output:05sequential_9/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02 
sequential_9/conv2d_33/BiasAdd?
sequential_9/conv2d_33/ReluRelu'sequential_9/conv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
sequential_9/conv2d_33/Relu?
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp?
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1?
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3)sequential_9/conv2d_33/Relu:activations:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<25
3sequential_9/batch_normalization_9/FusedBatchNormV3?
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_9/batch_normalization_9/AssignNewValue?
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_9/batch_normalization_9/AssignNewValue_1?
"sequential_9/up_sampling2d_4/ShapeShape7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2$
"sequential_9/up_sampling2d_4/Shape?
0sequential_9/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_9/up_sampling2d_4/strided_slice/stack?
2sequential_9/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_9/up_sampling2d_4/strided_slice/stack_1?
2sequential_9/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_9/up_sampling2d_4/strided_slice/stack_2?
*sequential_9/up_sampling2d_4/strided_sliceStridedSlice+sequential_9/up_sampling2d_4/Shape:output:09sequential_9/up_sampling2d_4/strided_slice/stack:output:0;sequential_9/up_sampling2d_4/strided_slice/stack_1:output:0;sequential_9/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_9/up_sampling2d_4/strided_slice?
"sequential_9/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_9/up_sampling2d_4/Const?
 sequential_9/up_sampling2d_4/mulMul3sequential_9/up_sampling2d_4/strided_slice:output:0+sequential_9/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2"
 sequential_9/up_sampling2d_4/mul?
9sequential_9/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0$sequential_9/up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2;
9sequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor?
,sequential_9/conv2d_34/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,sequential_9/conv2d_34/Conv2D/ReadVariableOp?
sequential_9/conv2d_34/Conv2DConv2DJsequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:04sequential_9/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
sequential_9/conv2d_34/Conv2D?
-sequential_9/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_9/conv2d_34/BiasAdd/ReadVariableOp?
sequential_9/conv2d_34/BiasAddBiasAdd&sequential_9/conv2d_34/Conv2D:output:05sequential_9/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
sequential_9/conv2d_34/BiasAdd?
sequential_9/conv2d_34/SigmoidSigmoid'sequential_9/conv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2 
sequential_9/conv2d_34/Sigmoid?
IdentityIdentity"sequential_9/conv2d_34/Sigmoid:y:02^sequential_8/batch_normalization_8/AssignNewValue4^sequential_8/batch_normalization_8/AssignNewValue_12^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2f
1sequential_8/batch_normalization_8/AssignNewValue1sequential_8/batch_normalization_8/AssignNewValue2j
3sequential_8/batch_normalization_8/AssignNewValue_13sequential_8/batch_normalization_8/AssignNewValue_12f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_1:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
5__inference_batch_normalization_9_layer_call_fn_35530

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_334252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????0::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_30_layer_call_and_return_conditional_losses_32918

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????H:::Y U
1
_output_shapes
:???????????H
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_33407

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35289

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????:::::j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_32843

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_28_layer_call_fn_35147

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_328642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_4_layer_call_fn_32849

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_328432
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?)
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_34741

inputs,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource,
(conv2d_30_conv2d_readvariableop_resource-
)conv2d_30_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dinputs'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_28/BiasAdd?
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_28/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2Dconv2d_28/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2
conv2d_29/BiasAdd?
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
conv2d_29/Relu?
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02!
conv2d_30/Conv2D/ReadVariableOp?
conv2d_30/Conv2DConv2Dconv2d_29/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_30/Conv2D?
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp?
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_30/BiasAdd?
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
conv2d_30/Relu?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_30/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
max_pooling2d_4/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*2
_output_shapes 
:????????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_4/MaxPool?
dropout_4/IdentityIdentity max_pooling2d_4/MaxPool:output:0*
T0*2
_output_shapes 
:????????????2
dropout_4/Identityz
IdentityIdentitydropout_4/Identity:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????:::::::::::Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32971

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3s
IdentityIdentityFusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????:::::Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_33025

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:????????????2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:????????????2

Identity_1"!

identity_1Identity_1:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_33020

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_9_layer_call_fn_34930

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_335612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_30_layer_call_and_return_conditional_losses_35178

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????H:::Y U
1
_output_shapes
:???????????H
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_34_layer_call_and_return_conditional_losses_33473

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????0:::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
b
)__inference_dropout_4_layer_call_fn_35337

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_330202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35486

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
??
?$
!__inference__traced_restore_35985
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate'
#assignvariableop_5_conv2d_28_kernel%
!assignvariableop_6_conv2d_28_bias'
#assignvariableop_7_conv2d_29_kernel%
!assignvariableop_8_conv2d_29_bias'
#assignvariableop_9_conv2d_30_kernel&
"assignvariableop_10_conv2d_30_bias3
/assignvariableop_11_batch_normalization_8_gamma2
.assignvariableop_12_batch_normalization_8_beta9
5assignvariableop_13_batch_normalization_8_moving_mean=
9assignvariableop_14_batch_normalization_8_moving_variance(
$assignvariableop_15_conv2d_31_kernel&
"assignvariableop_16_conv2d_31_bias(
$assignvariableop_17_conv2d_32_kernel&
"assignvariableop_18_conv2d_32_bias(
$assignvariableop_19_conv2d_33_kernel&
"assignvariableop_20_conv2d_33_bias3
/assignvariableop_21_batch_normalization_9_gamma2
.assignvariableop_22_batch_normalization_9_beta9
5assignvariableop_23_batch_normalization_9_moving_mean=
9assignvariableop_24_batch_normalization_9_moving_variance(
$assignvariableop_25_conv2d_34_kernel&
"assignvariableop_26_conv2d_34_bias
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1/
+assignvariableop_31_adam_conv2d_28_kernel_m-
)assignvariableop_32_adam_conv2d_28_bias_m/
+assignvariableop_33_adam_conv2d_29_kernel_m-
)assignvariableop_34_adam_conv2d_29_bias_m/
+assignvariableop_35_adam_conv2d_30_kernel_m-
)assignvariableop_36_adam_conv2d_30_bias_m:
6assignvariableop_37_adam_batch_normalization_8_gamma_m9
5assignvariableop_38_adam_batch_normalization_8_beta_m/
+assignvariableop_39_adam_conv2d_31_kernel_m-
)assignvariableop_40_adam_conv2d_31_bias_m/
+assignvariableop_41_adam_conv2d_32_kernel_m-
)assignvariableop_42_adam_conv2d_32_bias_m/
+assignvariableop_43_adam_conv2d_33_kernel_m-
)assignvariableop_44_adam_conv2d_33_bias_m:
6assignvariableop_45_adam_batch_normalization_9_gamma_m9
5assignvariableop_46_adam_batch_normalization_9_beta_m/
+assignvariableop_47_adam_conv2d_34_kernel_m-
)assignvariableop_48_adam_conv2d_34_bias_m/
+assignvariableop_49_adam_conv2d_28_kernel_v-
)assignvariableop_50_adam_conv2d_28_bias_v/
+assignvariableop_51_adam_conv2d_29_kernel_v-
)assignvariableop_52_adam_conv2d_29_bias_v/
+assignvariableop_53_adam_conv2d_30_kernel_v-
)assignvariableop_54_adam_conv2d_30_bias_v:
6assignvariableop_55_adam_batch_normalization_8_gamma_v9
5assignvariableop_56_adam_batch_normalization_8_beta_v/
+assignvariableop_57_adam_conv2d_31_kernel_v-
)assignvariableop_58_adam_conv2d_31_bias_v/
+assignvariableop_59_adam_conv2d_32_kernel_v-
)assignvariableop_60_adam_conv2d_32_bias_v/
+assignvariableop_61_adam_conv2d_33_kernel_v-
)assignvariableop_62_adam_conv2d_33_bias_v:
6assignvariableop_63_adam_batch_normalization_9_gamma_v9
5assignvariableop_64_adam_batch_normalization_9_beta_v/
+assignvariableop_65_adam_conv2d_34_kernel_v-
)assignvariableop_66_adam_conv2d_34_bias_v
identity_68??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_28_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d_28_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_29_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_29_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_30_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_30_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_8_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_8_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp5assignvariableop_13_batch_normalization_8_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp9assignvariableop_14_batch_normalization_8_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_31_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_31_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_32_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_32_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv2d_33_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_conv2d_33_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_9_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batch_normalization_9_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_9_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_9_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_conv2d_34_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_conv2d_34_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_28_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_28_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_29_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_29_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_30_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_30_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp6assignvariableop_37_adam_batch_normalization_8_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_batch_normalization_8_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_31_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_31_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_32_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_32_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_33_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_33_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_9_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_9_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_34_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_34_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_28_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_28_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_29_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_29_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_30_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_30_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_8_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_8_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_31_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_31_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_32_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_32_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_33_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_33_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_batch_normalization_9_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp5assignvariableop_64_adam_batch_normalization_9_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_34_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_34_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_669
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_67?
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_68"#
identity_68Identity_68:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?!
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_33102

inputs
conv2d_28_33075
conv2d_28_33077
conv2d_29_33080
conv2d_29_33082
conv2d_30_33085
conv2d_30_33087
batch_normalization_8_33090
batch_normalization_8_33092
batch_normalization_8_33094
batch_normalization_8_33096
identity??-batch_normalization_8/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_28_33075conv2d_28_33077*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_328642#
!conv2d_28/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_33080conv2d_29_33082*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_328912#
!conv2d_29/StatefulPartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_33085conv2d_30_33087*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_329182#
!conv2d_30/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0batch_normalization_8_33090batch_normalization_8_33092batch_normalization_8_33094batch_normalization_8_33096*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_329532/
-batch_normalization_8/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_328432!
max_pooling2d_4/PartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_330202#
!dropout_4/StatefulPartitionedCall?
IdentityIdentity*dropout_4/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_33_layer_call_fn_35402

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_333722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????H::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????H
 
_user_specified_nameinputs
ځ
?
__inference__traced_save_35774
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop/
+savev2_conv2d_30_kernel_read_readvariableop-
)savev2_conv2d_30_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_28_kernel_m_read_readvariableop4
0savev2_adam_conv2d_28_bias_m_read_readvariableop6
2savev2_adam_conv2d_29_kernel_m_read_readvariableop4
0savev2_adam_conv2d_29_bias_m_read_readvariableop6
2savev2_adam_conv2d_30_kernel_m_read_readvariableop4
0savev2_adam_conv2d_30_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_m_read_readvariableop6
2savev2_adam_conv2d_31_kernel_m_read_readvariableop4
0savev2_adam_conv2d_31_bias_m_read_readvariableop6
2savev2_adam_conv2d_32_kernel_m_read_readvariableop4
0savev2_adam_conv2d_32_bias_m_read_readvariableop6
2savev2_adam_conv2d_33_kernel_m_read_readvariableop4
0savev2_adam_conv2d_33_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_m_read_readvariableop6
2savev2_adam_conv2d_34_kernel_m_read_readvariableop4
0savev2_adam_conv2d_34_bias_m_read_readvariableop6
2savev2_adam_conv2d_28_kernel_v_read_readvariableop4
0savev2_adam_conv2d_28_bias_v_read_readvariableop6
2savev2_adam_conv2d_29_kernel_v_read_readvariableop4
0savev2_adam_conv2d_29_bias_v_read_readvariableop6
2savev2_adam_conv2d_30_kernel_v_read_readvariableop4
0savev2_adam_conv2d_30_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_v_read_readvariableop6
2savev2_adam_conv2d_31_kernel_v_read_readvariableop4
0savev2_adam_conv2d_31_bias_v_read_readvariableop6
2savev2_adam_conv2d_32_kernel_v_read_readvariableop4
0savev2_adam_conv2d_32_bias_v_read_readvariableop6
2savev2_adam_conv2d_33_kernel_v_read_readvariableop4
0savev2_adam_conv2d_33_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_v_read_readvariableop6
2savev2_adam_conv2d_34_kernel_v_read_readvariableop4
0savev2_adam_conv2d_34_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6cf0b99c707f4c469d14a1f7a6b57113/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop+savev2_conv2d_30_kernel_read_readvariableop)savev2_conv2d_30_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_28_kernel_m_read_readvariableop0savev2_adam_conv2d_28_bias_m_read_readvariableop2savev2_adam_conv2d_29_kernel_m_read_readvariableop0savev2_adam_conv2d_29_bias_m_read_readvariableop2savev2_adam_conv2d_30_kernel_m_read_readvariableop0savev2_adam_conv2d_30_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop2savev2_adam_conv2d_31_kernel_m_read_readvariableop0savev2_adam_conv2d_31_bias_m_read_readvariableop2savev2_adam_conv2d_32_kernel_m_read_readvariableop0savev2_adam_conv2d_32_bias_m_read_readvariableop2savev2_adam_conv2d_33_kernel_m_read_readvariableop0savev2_adam_conv2d_33_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop2savev2_adam_conv2d_34_kernel_m_read_readvariableop0savev2_adam_conv2d_34_bias_m_read_readvariableop2savev2_adam_conv2d_28_kernel_v_read_readvariableop0savev2_adam_conv2d_28_bias_v_read_readvariableop2savev2_adam_conv2d_29_kernel_v_read_readvariableop0savev2_adam_conv2d_29_bias_v_read_readvariableop2savev2_adam_conv2d_30_kernel_v_read_readvariableop0savev2_adam_conv2d_30_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop2savev2_adam_conv2d_31_kernel_v_read_readvariableop0savev2_adam_conv2d_31_bias_v_read_readvariableop2savev2_adam_conv2d_32_kernel_v_read_readvariableop0savev2_adam_conv2d_32_bias_v_read_readvariableop2savev2_adam_conv2d_33_kernel_v_read_readvariableop0savev2_adam_conv2d_33_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop2savev2_adam_conv2d_34_kernel_v_read_readvariableop0savev2_adam_conv2d_34_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :0:0:0H:H:H?:?:?:?:?:?:??:?:?H:H:H0:0:0:0:0:0:0:: : : : :0:0:0H:H:H?:?:?:?:??:?:?H:H:H0:0:0:0:0::0:0:0H:H:H?:?:?:?:??:?:?H:H:H0:0:0:0:0:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0H: 	

_output_shapes
:H:-
)
'
_output_shapes
:H?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:-)
'
_output_shapes
:?H: 

_output_shapes
:H:,(
&
_output_shapes
:H0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
:0: !

_output_shapes
:0:,"(
&
_output_shapes
:0H: #

_output_shapes
:H:-$)
'
_output_shapes
:H?:!%

_output_shapes	
:?:!&

_output_shapes	
:?:!'

_output_shapes	
:?:.(*
(
_output_shapes
:??:!)

_output_shapes	
:?:-*)
'
_output_shapes
:?H: +

_output_shapes
:H:,,(
&
_output_shapes
:H0: -

_output_shapes
:0: .

_output_shapes
:0: /

_output_shapes
:0:,0(
&
_output_shapes
:0: 1

_output_shapes
::,2(
&
_output_shapes
:0: 3

_output_shapes
:0:,4(
&
_output_shapes
:0H: 5

_output_shapes
:H:-6)
'
_output_shapes
:H?:!7

_output_shapes	
:?:!8

_output_shapes	
:?:!9

_output_shapes	
:?:.:*
(
_output_shapes
:??:!;

_output_shapes	
:?:-<)
'
_output_shapes
:?H: =

_output_shapes
:H:,>(
&
_output_shapes
:H0: ?

_output_shapes
:0: @

_output_shapes
:0: A

_output_shapes
:0:,B(
&
_output_shapes
:0: C

_output_shapes
::D

_output_shapes
: 
?	
?
,__inference_sequential_8_layer_call_fn_34766

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_331022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_33913
x
sequential_8_33866
sequential_8_33868
sequential_8_33870
sequential_8_33872
sequential_8_33874
sequential_8_33876
sequential_8_33878
sequential_8_33880
sequential_8_33882
sequential_8_33884
sequential_9_33887
sequential_9_33889
sequential_9_33891
sequential_9_33893
sequential_9_33895
sequential_9_33897
sequential_9_33899
sequential_9_33901
sequential_9_33903
sequential_9_33905
sequential_9_33907
sequential_9_33909
identity??$sequential_8/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallxsequential_8_33866sequential_8_33868sequential_8_33870sequential_8_33872sequential_8_33874sequential_8_33876sequential_8_33878sequential_8_33880sequential_8_33882sequential_8_33884*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_331572&
$sequential_8/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0sequential_9_33887sequential_9_33889sequential_9_33891sequential_9_33893sequential_9_33895sequential_9_33897sequential_9_33899sequential_9_33901sequential_9_33903sequential_9_33905sequential_9_33907sequential_9_33909*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_336242&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_33069
input_5
conv2d_28_33042
conv2d_28_33044
conv2d_29_33047
conv2d_29_33049
conv2d_30_33052
conv2d_30_33054
batch_normalization_8_33057
batch_normalization_8_33059
batch_normalization_8_33061
batch_normalization_8_33063
identity??-batch_normalization_8/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_28_33042conv2d_28_33044*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_328642#
!conv2d_28/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_33047conv2d_29_33049*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_328912#
!conv2d_29/StatefulPartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_33052conv2d_30_33054*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_329182#
!conv2d_30/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0batch_normalization_8_33057batch_normalization_8_33059batch_normalization_8_33061batch_normalization_8_33063*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_329712/
-batch_normalization_8/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_328432!
max_pooling2d_4/PartitionedCall?
dropout_4/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_330252
dropout_4/PartitionedCall?
IdentityIdentity"dropout_4/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?:
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_34901

inputs,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource,
(conv2d_32_conv2d_readvariableop_resource-
)conv2d_32_biasadd_readvariableop_resource,
(conv2d_33_conv2d_readvariableop_resource-
)conv2d_33_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource
identity??
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_31/Conv2D/ReadVariableOp?
conv2d_31/Conv2DConv2Dinputs'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_31/Conv2D?
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp?
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_31/BiasAdd?
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
conv2d_31/Relu?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02!
conv2d_32/Conv2D/ReadVariableOp?
conv2d_32/Conv2DConv2Dconv2d_31/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
conv2d_32/Conv2D?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2
conv2d_32/BiasAdd?
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
conv2d_32/Relu?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02!
conv2d_33/Conv2D/ReadVariableOp?
conv2d_33/Conv2DConv2Dconv2d_32/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_33/Conv2D?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_33/BiasAdd?
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_33/Relu?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_33/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3?
up_sampling2d_4/ShapeShape*batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape?
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack?
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1?
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2?
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor*batch_normalization_9/FusedBatchNormV3:y:0up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_34/BiasAdd?
conv2d_34/SigmoidSigmoidconv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_34/Sigmoids
IdentityIdentityconv2d_34/Sigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????:::::::::::::Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_33425

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????0:::::Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?<
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_34700

inputs,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource,
(conv2d_30_conv2d_readvariableop_resource-
)conv2d_30_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dinputs'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_28/BiasAdd?
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_28/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2Dconv2d_28/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2
conv2d_29/BiasAdd?
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
conv2d_29/Relu?
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02!
conv2d_30/Conv2D/ReadVariableOp?
conv2d_30/Conv2DConv2Dconv2d_29/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_30/Conv2D?
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp?
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_30/BiasAdd?
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
conv2d_30/Relu?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_30/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_8/FusedBatchNormV3?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1?
max_pooling2d_4/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*2
_output_shapes 
:????????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_4/MaxPoolw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_4/dropout/Const?
dropout_4/dropout/MulMul max_pooling2d_4/MaxPool:output:0 dropout_4/dropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeShape max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout_4/dropout/Mul_1?
IdentityIdentitydropout_4/dropout/Mul_1:z:0%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_29_layer_call_fn_35167

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_328912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????H2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????0::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_8_layer_call_fn_35315

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_328262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_8_layer_call_fn_35238

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_329532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
7__inference_denoising_autoencoder_4_layer_call_fn_34310
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_339132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32826

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????:::::j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_33039
input_5
conv2d_28_32875
conv2d_28_32877
conv2d_29_32902
conv2d_29_32904
conv2d_30_32929
conv2d_30_32931
batch_normalization_8_32998
batch_normalization_8_33000
batch_normalization_8_33002
batch_normalization_8_33004
identity??-batch_normalization_8/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_28_32875conv2d_28_32877*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_328642#
!conv2d_28/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_32902conv2d_29_32904*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_328912#
!conv2d_29/StatefulPartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_32929conv2d_30_32931*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_329182#
!conv2d_30/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0batch_normalization_8_32998batch_normalization_8_33000batch_normalization_8_33002batch_normalization_8_33004*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_329532/
-batch_normalization_8/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_328432!
max_pooling2d_4/PartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_330202#
!dropout_4/StatefulPartitionedCall?
IdentityIdentity*dropout_4/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?
?
5__inference_batch_normalization_9_layer_call_fn_35517

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_334072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????0::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32953

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?"
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_33624

inputs
conv2d_31_33593
conv2d_31_33595
conv2d_32_33598
conv2d_32_33600
conv2d_33_33603
conv2d_33_33605
batch_normalization_9_33608
batch_normalization_9_33610
batch_normalization_9_33612
batch_normalization_9_33614
conv2d_34_33618
conv2d_34_33620
identity??-batch_normalization_9/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_31_33593conv2d_31_33595*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_333182#
!conv2d_31/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_33598conv2d_32_33600*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_333452#
!conv2d_32/StatefulPartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0conv2d_33_33603conv2d_33_33605*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_333722#
!conv2d_33/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0batch_normalization_9_33608batch_normalization_9_33610batch_normalization_9_33612batch_normalization_9_33614*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_334252/
-batch_normalization_9/StatefulPartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_332972!
up_sampling2d_4/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_34_33618conv2d_34_33620*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_334732#
!conv2d_34/StatefulPartitionedCall?
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????::::::::::::2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_33_layer_call_and_return_conditional_losses_35393

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????H:::Y U
1
_output_shapes
:???????????H
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_32_layer_call_and_return_conditional_losses_33345

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????H2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????:::Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_34_layer_call_fn_35550

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_334732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????0::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_32891

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????H2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????0:::Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_35158

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????H2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????0:::Y U
1
_output_shapes
:???????????0
 
_user_specified_nameinputs
?

?
,__inference_sequential_9_layer_call_fn_35098
conv2d_31_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_335612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:c _
2
_output_shapes 
:????????????
)
_user_specified_nameconv2d_31_input
?
K
/__inference_up_sampling2d_4_layer_call_fn_33303

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_332972
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
,__inference_sequential_9_layer_call_fn_35127
conv2d_31_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_336242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:c _
2
_output_shapes 
:????????????
)
_user_specified_nameconv2d_31_input
?"
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_33561

inputs
conv2d_31_33530
conv2d_31_33532
conv2d_32_33535
conv2d_32_33537
conv2d_33_33540
conv2d_33_33542
batch_normalization_9_33545
batch_normalization_9_33547
batch_normalization_9_33549
batch_normalization_9_33551
conv2d_34_33555
conv2d_34_33557
identity??-batch_normalization_9/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_31_33530conv2d_31_33532*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_333182#
!conv2d_31/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_33535conv2d_32_33537*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_333452#
!conv2d_32/StatefulPartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0conv2d_33_33540conv2d_33_33542*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_333722#
!conv2d_33/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0batch_normalization_9_33545batch_normalization_9_33547batch_normalization_9_33549batch_normalization_9_33551*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_334072/
-batch_normalization_9/StatefulPartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_332972!
up_sampling2d_4/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_34_33555conv2d_34_33557*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_334732#
!conv2d_34/StatefulPartitionedCall?
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????::::::::::::2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_34_layer_call_and_return_conditional_losses_35541

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????0:::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
,__inference_sequential_8_layer_call_fn_34791

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_331572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_35332

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:????????????2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:????????????2

Identity_1"!

identity_1Identity_1:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_32_layer_call_and_return_conditional_losses_35373

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????H2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????:::Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_31_layer_call_fn_35362

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_333182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_33242

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_33157

inputs
conv2d_28_33130
conv2d_28_33132
conv2d_29_33135
conv2d_29_33137
conv2d_30_33140
conv2d_30_33142
batch_normalization_8_33145
batch_normalization_8_33147
batch_normalization_8_33149
batch_normalization_8_33151
identity??-batch_normalization_8/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_28_33130conv2d_28_33132*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_328642#
!conv2d_28/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_33135conv2d_29_33137*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_328912#
!conv2d_29/StatefulPartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_33140conv2d_30_33142*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_329182#
!conv2d_30/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0batch_normalization_8_33145batch_normalization_8_33147batch_normalization_8_33149batch_normalization_8_33151*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_329712/
-batch_normalization_8/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_328432!
max_pooling2d_4/PartitionedCall?
dropout_4/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_330252
dropout_4/PartitionedCall?
IdentityIdentity"dropout_4/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_33273

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????0:::::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
~
)__inference_conv2d_30_layer_call_fn_35187

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_329182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????H::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????H
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_35327

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
??
?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34170
input_19
5sequential_8_conv2d_28_conv2d_readvariableop_resource:
6sequential_8_conv2d_28_biasadd_readvariableop_resource9
5sequential_8_conv2d_29_conv2d_readvariableop_resource:
6sequential_8_conv2d_29_biasadd_readvariableop_resource9
5sequential_8_conv2d_30_conv2d_readvariableop_resource:
6sequential_8_conv2d_30_biasadd_readvariableop_resource>
:sequential_8_batch_normalization_8_readvariableop_resource@
<sequential_8_batch_normalization_8_readvariableop_1_resourceO
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceQ
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource9
5sequential_9_conv2d_31_conv2d_readvariableop_resource:
6sequential_9_conv2d_31_biasadd_readvariableop_resource9
5sequential_9_conv2d_32_conv2d_readvariableop_resource:
6sequential_9_conv2d_32_biasadd_readvariableop_resource9
5sequential_9_conv2d_33_conv2d_readvariableop_resource:
6sequential_9_conv2d_33_biasadd_readvariableop_resource>
:sequential_9_batch_normalization_9_readvariableop_resource@
<sequential_9_batch_normalization_9_readvariableop_1_resourceO
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceQ
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource9
5sequential_9_conv2d_34_conv2d_readvariableop_resource:
6sequential_9_conv2d_34_biasadd_readvariableop_resource
identity??1sequential_8/batch_normalization_8/AssignNewValue?3sequential_8/batch_normalization_8/AssignNewValue_1?1sequential_9/batch_normalization_9/AssignNewValue?3sequential_9/batch_normalization_9/AssignNewValue_1?
,sequential_8/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,sequential_8/conv2d_28/Conv2D/ReadVariableOp?
sequential_8/conv2d_28/Conv2DConv2Dinput_14sequential_8/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
sequential_8/conv2d_28/Conv2D?
-sequential_8/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_8/conv2d_28/BiasAdd/ReadVariableOp?
sequential_8/conv2d_28/BiasAddBiasAdd&sequential_8/conv2d_28/Conv2D:output:05sequential_8/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02 
sequential_8/conv2d_28/BiasAdd?
sequential_8/conv2d_28/ReluRelu'sequential_8/conv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
sequential_8/conv2d_28/Relu?
,sequential_8/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02.
,sequential_8/conv2d_29/Conv2D/ReadVariableOp?
sequential_8/conv2d_29/Conv2DConv2D)sequential_8/conv2d_28/Relu:activations:04sequential_8/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
sequential_8/conv2d_29/Conv2D?
-sequential_8/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_8/conv2d_29/BiasAdd/ReadVariableOp?
sequential_8/conv2d_29/BiasAddBiasAdd&sequential_8/conv2d_29/Conv2D:output:05sequential_8/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2 
sequential_8/conv2d_29/BiasAdd?
sequential_8/conv2d_29/ReluRelu'sequential_8/conv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
sequential_8/conv2d_29/Relu?
,sequential_8/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_30_conv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02.
,sequential_8/conv2d_30/Conv2D/ReadVariableOp?
sequential_8/conv2d_30/Conv2DConv2D)sequential_8/conv2d_29/Relu:activations:04sequential_8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential_8/conv2d_30/Conv2D?
-sequential_8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_30/BiasAdd/ReadVariableOp?
sequential_8/conv2d_30/BiasAddBiasAdd&sequential_8/conv2d_30/Conv2D:output:05sequential_8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2 
sequential_8/conv2d_30/BiasAdd?
sequential_8/conv2d_30/ReluRelu'sequential_8/conv2d_30/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential_8/conv2d_30/Relu?
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp?
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1?
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3)sequential_8/conv2d_30/Relu:activations:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<25
3sequential_8/batch_normalization_8/FusedBatchNormV3?
1sequential_8/batch_normalization_8/AssignNewValueAssignVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource@sequential_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_8/batch_normalization_8/AssignNewValue?
3sequential_8/batch_normalization_8/AssignNewValue_1AssignVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceDsequential_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0E^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_8/batch_normalization_8/AssignNewValue_1?
$sequential_8/max_pooling2d_4/MaxPoolMaxPool7sequential_8/batch_normalization_8/FusedBatchNormV3:y:0*2
_output_shapes 
:????????????*
ksize
*
paddingSAME*
strides
2&
$sequential_8/max_pooling2d_4/MaxPool?
$sequential_8/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential_8/dropout_4/dropout/Const?
"sequential_8/dropout_4/dropout/MulMul-sequential_8/max_pooling2d_4/MaxPool:output:0-sequential_8/dropout_4/dropout/Const:output:0*
T0*2
_output_shapes 
:????????????2$
"sequential_8/dropout_4/dropout/Mul?
$sequential_8/dropout_4/dropout/ShapeShape-sequential_8/max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
:2&
$sequential_8/dropout_4/dropout/Shape?
;sequential_8/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_8/dropout_4/dropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype02=
;sequential_8/dropout_4/dropout/random_uniform/RandomUniform?
-sequential_8/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential_8/dropout_4/dropout/GreaterEqual/y?
+sequential_8/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_8/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_8/dropout_4/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2-
+sequential_8/dropout_4/dropout/GreaterEqual?
#sequential_8/dropout_4/dropout/CastCast/sequential_8/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2%
#sequential_8/dropout_4/dropout/Cast?
$sequential_8/dropout_4/dropout/Mul_1Mul&sequential_8/dropout_4/dropout/Mul:z:0'sequential_8/dropout_4/dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2&
$sequential_8/dropout_4/dropout/Mul_1?
,sequential_9/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,sequential_9/conv2d_31/Conv2D/ReadVariableOp?
sequential_9/conv2d_31/Conv2DConv2D(sequential_8/dropout_4/dropout/Mul_1:z:04sequential_9/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential_9/conv2d_31/Conv2D?
-sequential_9/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_9/conv2d_31/BiasAdd/ReadVariableOp?
sequential_9/conv2d_31/BiasAddBiasAdd&sequential_9/conv2d_31/Conv2D:output:05sequential_9/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2 
sequential_9/conv2d_31/BiasAdd?
sequential_9/conv2d_31/ReluRelu'sequential_9/conv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential_9/conv2d_31/Relu?
,sequential_9/conv2d_32/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02.
,sequential_9/conv2d_32/Conv2D/ReadVariableOp?
sequential_9/conv2d_32/Conv2DConv2D)sequential_9/conv2d_31/Relu:activations:04sequential_9/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
sequential_9/conv2d_32/Conv2D?
-sequential_9/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_9/conv2d_32/BiasAdd/ReadVariableOp?
sequential_9/conv2d_32/BiasAddBiasAdd&sequential_9/conv2d_32/Conv2D:output:05sequential_9/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2 
sequential_9/conv2d_32/BiasAdd?
sequential_9/conv2d_32/ReluRelu'sequential_9/conv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
sequential_9/conv2d_32/Relu?
,sequential_9/conv2d_33/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02.
,sequential_9/conv2d_33/Conv2D/ReadVariableOp?
sequential_9/conv2d_33/Conv2DConv2D)sequential_9/conv2d_32/Relu:activations:04sequential_9/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
sequential_9/conv2d_33/Conv2D?
-sequential_9/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_9/conv2d_33/BiasAdd/ReadVariableOp?
sequential_9/conv2d_33/BiasAddBiasAdd&sequential_9/conv2d_33/Conv2D:output:05sequential_9/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02 
sequential_9/conv2d_33/BiasAdd?
sequential_9/conv2d_33/ReluRelu'sequential_9/conv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
sequential_9/conv2d_33/Relu?
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp?
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1?
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3)sequential_9/conv2d_33/Relu:activations:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<25
3sequential_9/batch_normalization_9/FusedBatchNormV3?
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_9/batch_normalization_9/AssignNewValue?
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_9/batch_normalization_9/AssignNewValue_1?
"sequential_9/up_sampling2d_4/ShapeShape7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2$
"sequential_9/up_sampling2d_4/Shape?
0sequential_9/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_9/up_sampling2d_4/strided_slice/stack?
2sequential_9/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_9/up_sampling2d_4/strided_slice/stack_1?
2sequential_9/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_9/up_sampling2d_4/strided_slice/stack_2?
*sequential_9/up_sampling2d_4/strided_sliceStridedSlice+sequential_9/up_sampling2d_4/Shape:output:09sequential_9/up_sampling2d_4/strided_slice/stack:output:0;sequential_9/up_sampling2d_4/strided_slice/stack_1:output:0;sequential_9/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_9/up_sampling2d_4/strided_slice?
"sequential_9/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_9/up_sampling2d_4/Const?
 sequential_9/up_sampling2d_4/mulMul3sequential_9/up_sampling2d_4/strided_slice:output:0+sequential_9/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2"
 sequential_9/up_sampling2d_4/mul?
9sequential_9/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0$sequential_9/up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2;
9sequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor?
,sequential_9/conv2d_34/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,sequential_9/conv2d_34/Conv2D/ReadVariableOp?
sequential_9/conv2d_34/Conv2DConv2DJsequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:04sequential_9/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
sequential_9/conv2d_34/Conv2D?
-sequential_9/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_9/conv2d_34/BiasAdd/ReadVariableOp?
sequential_9/conv2d_34/BiasAddBiasAdd&sequential_9/conv2d_34/Conv2D:output:05sequential_9/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
sequential_9/conv2d_34/BiasAdd?
sequential_9/conv2d_34/SigmoidSigmoid'sequential_9/conv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2 
sequential_9/conv2d_34/Sigmoid?
IdentityIdentity"sequential_9/conv2d_34/Sigmoid:y:02^sequential_8/batch_normalization_8/AssignNewValue4^sequential_8/batch_normalization_8/AssignNewValue_12^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::2f
1sequential_8/batch_normalization_8/AssignNewValue1sequential_8/batch_normalization_8/AssignNewValue2j
3sequential_8/batch_normalization_8/AssignNewValue_13sequential_8/batch_normalization_8/AssignNewValue_12f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_1:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?	
?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_32864

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????02
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????02

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_9_layer_call_fn_34959

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_336242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_31_layer_call_and_return_conditional_losses_35353

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????:::Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35440

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????0:::::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?v
?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34552
x9
5sequential_8_conv2d_28_conv2d_readvariableop_resource:
6sequential_8_conv2d_28_biasadd_readvariableop_resource9
5sequential_8_conv2d_29_conv2d_readvariableop_resource:
6sequential_8_conv2d_29_biasadd_readvariableop_resource9
5sequential_8_conv2d_30_conv2d_readvariableop_resource:
6sequential_8_conv2d_30_biasadd_readvariableop_resource>
:sequential_8_batch_normalization_8_readvariableop_resource@
<sequential_8_batch_normalization_8_readvariableop_1_resourceO
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceQ
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource9
5sequential_9_conv2d_31_conv2d_readvariableop_resource:
6sequential_9_conv2d_31_biasadd_readvariableop_resource9
5sequential_9_conv2d_32_conv2d_readvariableop_resource:
6sequential_9_conv2d_32_biasadd_readvariableop_resource9
5sequential_9_conv2d_33_conv2d_readvariableop_resource:
6sequential_9_conv2d_33_biasadd_readvariableop_resource>
:sequential_9_batch_normalization_9_readvariableop_resource@
<sequential_9_batch_normalization_9_readvariableop_1_resourceO
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceQ
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource9
5sequential_9_conv2d_34_conv2d_readvariableop_resource:
6sequential_9_conv2d_34_biasadd_readvariableop_resource
identity??
,sequential_8/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,sequential_8/conv2d_28/Conv2D/ReadVariableOp?
sequential_8/conv2d_28/Conv2DConv2Dx4sequential_8/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
sequential_8/conv2d_28/Conv2D?
-sequential_8/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_8/conv2d_28/BiasAdd/ReadVariableOp?
sequential_8/conv2d_28/BiasAddBiasAdd&sequential_8/conv2d_28/Conv2D:output:05sequential_8/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02 
sequential_8/conv2d_28/BiasAdd?
sequential_8/conv2d_28/ReluRelu'sequential_8/conv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
sequential_8/conv2d_28/Relu?
,sequential_8/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02.
,sequential_8/conv2d_29/Conv2D/ReadVariableOp?
sequential_8/conv2d_29/Conv2DConv2D)sequential_8/conv2d_28/Relu:activations:04sequential_8/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
sequential_8/conv2d_29/Conv2D?
-sequential_8/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_8/conv2d_29/BiasAdd/ReadVariableOp?
sequential_8/conv2d_29/BiasAddBiasAdd&sequential_8/conv2d_29/Conv2D:output:05sequential_8/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2 
sequential_8/conv2d_29/BiasAdd?
sequential_8/conv2d_29/ReluRelu'sequential_8/conv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
sequential_8/conv2d_29/Relu?
,sequential_8/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_30_conv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02.
,sequential_8/conv2d_30/Conv2D/ReadVariableOp?
sequential_8/conv2d_30/Conv2DConv2D)sequential_8/conv2d_29/Relu:activations:04sequential_8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential_8/conv2d_30/Conv2D?
-sequential_8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_8/conv2d_30/BiasAdd/ReadVariableOp?
sequential_8/conv2d_30/BiasAddBiasAdd&sequential_8/conv2d_30/Conv2D:output:05sequential_8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2 
sequential_8/conv2d_30/BiasAdd?
sequential_8/conv2d_30/ReluRelu'sequential_8/conv2d_30/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential_8/conv2d_30/Relu?
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOp?
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1?
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3)sequential_8/conv2d_30/Relu:activations:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3?
$sequential_8/max_pooling2d_4/MaxPoolMaxPool7sequential_8/batch_normalization_8/FusedBatchNormV3:y:0*2
_output_shapes 
:????????????*
ksize
*
paddingSAME*
strides
2&
$sequential_8/max_pooling2d_4/MaxPool?
sequential_8/dropout_4/IdentityIdentity-sequential_8/max_pooling2d_4/MaxPool:output:0*
T0*2
_output_shapes 
:????????????2!
sequential_8/dropout_4/Identity?
,sequential_9/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,sequential_9/conv2d_31/Conv2D/ReadVariableOp?
sequential_9/conv2d_31/Conv2DConv2D(sequential_8/dropout_4/Identity:output:04sequential_9/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
sequential_9/conv2d_31/Conv2D?
-sequential_9/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_9/conv2d_31/BiasAdd/ReadVariableOp?
sequential_9/conv2d_31/BiasAddBiasAdd&sequential_9/conv2d_31/Conv2D:output:05sequential_9/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2 
sequential_9/conv2d_31/BiasAdd?
sequential_9/conv2d_31/ReluRelu'sequential_9/conv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
sequential_9/conv2d_31/Relu?
,sequential_9/conv2d_32/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02.
,sequential_9/conv2d_32/Conv2D/ReadVariableOp?
sequential_9/conv2d_32/Conv2DConv2D)sequential_9/conv2d_31/Relu:activations:04sequential_9/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
sequential_9/conv2d_32/Conv2D?
-sequential_9/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_9/conv2d_32/BiasAdd/ReadVariableOp?
sequential_9/conv2d_32/BiasAddBiasAdd&sequential_9/conv2d_32/Conv2D:output:05sequential_9/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2 
sequential_9/conv2d_32/BiasAdd?
sequential_9/conv2d_32/ReluRelu'sequential_9/conv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
sequential_9/conv2d_32/Relu?
,sequential_9/conv2d_33/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02.
,sequential_9/conv2d_33/Conv2D/ReadVariableOp?
sequential_9/conv2d_33/Conv2DConv2D)sequential_9/conv2d_32/Relu:activations:04sequential_9/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
sequential_9/conv2d_33/Conv2D?
-sequential_9/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_9/conv2d_33/BiasAdd/ReadVariableOp?
sequential_9/conv2d_33/BiasAddBiasAdd&sequential_9/conv2d_33/Conv2D:output:05sequential_9/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02 
sequential_9/conv2d_33/BiasAdd?
sequential_9/conv2d_33/ReluRelu'sequential_9/conv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
sequential_9/conv2d_33/Relu?
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype023
1sequential_9/batch_normalization_9/ReadVariableOp?
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype025
3sequential_9/batch_normalization_9/ReadVariableOp_1?
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02D
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02F
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3)sequential_9/conv2d_33/Relu:activations:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 25
3sequential_9/batch_normalization_9/FusedBatchNormV3?
"sequential_9/up_sampling2d_4/ShapeShape7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2$
"sequential_9/up_sampling2d_4/Shape?
0sequential_9/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_9/up_sampling2d_4/strided_slice/stack?
2sequential_9/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_9/up_sampling2d_4/strided_slice/stack_1?
2sequential_9/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_9/up_sampling2d_4/strided_slice/stack_2?
*sequential_9/up_sampling2d_4/strided_sliceStridedSlice+sequential_9/up_sampling2d_4/Shape:output:09sequential_9/up_sampling2d_4/strided_slice/stack:output:0;sequential_9/up_sampling2d_4/strided_slice/stack_1:output:0;sequential_9/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_9/up_sampling2d_4/strided_slice?
"sequential_9/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_9/up_sampling2d_4/Const?
 sequential_9/up_sampling2d_4/mulMul3sequential_9/up_sampling2d_4/strided_slice:output:0+sequential_9/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2"
 sequential_9/up_sampling2d_4/mul?
9sequential_9/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0$sequential_9/up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2;
9sequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor?
,sequential_9/conv2d_34/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,sequential_9/conv2d_34/Conv2D/ReadVariableOp?
sequential_9/conv2d_34/Conv2DConv2DJsequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:04sequential_9/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
sequential_9/conv2d_34/Conv2D?
-sequential_9/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_9/conv2d_34/BiasAdd/ReadVariableOp?
sequential_9/conv2d_34/BiasAddBiasAdd&sequential_9/conv2d_34/Conv2D:output:05sequential_9/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
sequential_9/conv2d_34/BiasAdd?
sequential_9/conv2d_34/SigmoidSigmoid'sequential_9/conv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2 
sequential_9/conv2d_34/Sigmoid?
IdentityIdentity"sequential_9/conv2d_34/Sigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????:::::::::::::::::::::::T P
1
_output_shapes
:???????????

_user_specified_namex
??
?
 __inference__wrapped_model_32733
input_1Q
Mdenoising_autoencoder_4_sequential_8_conv2d_28_conv2d_readvariableop_resourceR
Ndenoising_autoencoder_4_sequential_8_conv2d_28_biasadd_readvariableop_resourceQ
Mdenoising_autoencoder_4_sequential_8_conv2d_29_conv2d_readvariableop_resourceR
Ndenoising_autoencoder_4_sequential_8_conv2d_29_biasadd_readvariableop_resourceQ
Mdenoising_autoencoder_4_sequential_8_conv2d_30_conv2d_readvariableop_resourceR
Ndenoising_autoencoder_4_sequential_8_conv2d_30_biasadd_readvariableop_resourceV
Rdenoising_autoencoder_4_sequential_8_batch_normalization_8_readvariableop_resourceX
Tdenoising_autoencoder_4_sequential_8_batch_normalization_8_readvariableop_1_resourceg
cdenoising_autoencoder_4_sequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourcei
edenoising_autoencoder_4_sequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceQ
Mdenoising_autoencoder_4_sequential_9_conv2d_31_conv2d_readvariableop_resourceR
Ndenoising_autoencoder_4_sequential_9_conv2d_31_biasadd_readvariableop_resourceQ
Mdenoising_autoencoder_4_sequential_9_conv2d_32_conv2d_readvariableop_resourceR
Ndenoising_autoencoder_4_sequential_9_conv2d_32_biasadd_readvariableop_resourceQ
Mdenoising_autoencoder_4_sequential_9_conv2d_33_conv2d_readvariableop_resourceR
Ndenoising_autoencoder_4_sequential_9_conv2d_33_biasadd_readvariableop_resourceV
Rdenoising_autoencoder_4_sequential_9_batch_normalization_9_readvariableop_resourceX
Tdenoising_autoencoder_4_sequential_9_batch_normalization_9_readvariableop_1_resourceg
cdenoising_autoencoder_4_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resourcei
edenoising_autoencoder_4_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceQ
Mdenoising_autoencoder_4_sequential_9_conv2d_34_conv2d_readvariableop_resourceR
Ndenoising_autoencoder_4_sequential_9_conv2d_34_biasadd_readvariableop_resource
identity??
Ddenoising_autoencoder_4/sequential_8/conv2d_28/Conv2D/ReadVariableOpReadVariableOpMdenoising_autoencoder_4_sequential_8_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02F
Ddenoising_autoencoder_4/sequential_8/conv2d_28/Conv2D/ReadVariableOp?
5denoising_autoencoder_4/sequential_8/conv2d_28/Conv2DConv2Dinput_1Ldenoising_autoencoder_4/sequential_8/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
27
5denoising_autoencoder_4/sequential_8/conv2d_28/Conv2D?
Edenoising_autoencoder_4/sequential_8/conv2d_28/BiasAdd/ReadVariableOpReadVariableOpNdenoising_autoencoder_4_sequential_8_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02G
Edenoising_autoencoder_4/sequential_8/conv2d_28/BiasAdd/ReadVariableOp?
6denoising_autoencoder_4/sequential_8/conv2d_28/BiasAddBiasAdd>denoising_autoencoder_4/sequential_8/conv2d_28/Conv2D:output:0Mdenoising_autoencoder_4/sequential_8/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????028
6denoising_autoencoder_4/sequential_8/conv2d_28/BiasAdd?
3denoising_autoencoder_4/sequential_8/conv2d_28/ReluRelu?denoising_autoencoder_4/sequential_8/conv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:???????????025
3denoising_autoencoder_4/sequential_8/conv2d_28/Relu?
Ddenoising_autoencoder_4/sequential_8/conv2d_29/Conv2D/ReadVariableOpReadVariableOpMdenoising_autoencoder_4_sequential_8_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:0H*
dtype02F
Ddenoising_autoencoder_4/sequential_8/conv2d_29/Conv2D/ReadVariableOp?
5denoising_autoencoder_4/sequential_8/conv2d_29/Conv2DConv2DAdenoising_autoencoder_4/sequential_8/conv2d_28/Relu:activations:0Ldenoising_autoencoder_4/sequential_8/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
27
5denoising_autoencoder_4/sequential_8/conv2d_29/Conv2D?
Edenoising_autoencoder_4/sequential_8/conv2d_29/BiasAdd/ReadVariableOpReadVariableOpNdenoising_autoencoder_4_sequential_8_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02G
Edenoising_autoencoder_4/sequential_8/conv2d_29/BiasAdd/ReadVariableOp?
6denoising_autoencoder_4/sequential_8/conv2d_29/BiasAddBiasAdd>denoising_autoencoder_4/sequential_8/conv2d_29/Conv2D:output:0Mdenoising_autoencoder_4/sequential_8/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H28
6denoising_autoencoder_4/sequential_8/conv2d_29/BiasAdd?
3denoising_autoencoder_4/sequential_8/conv2d_29/ReluRelu?denoising_autoencoder_4/sequential_8/conv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H25
3denoising_autoencoder_4/sequential_8/conv2d_29/Relu?
Ddenoising_autoencoder_4/sequential_8/conv2d_30/Conv2D/ReadVariableOpReadVariableOpMdenoising_autoencoder_4_sequential_8_conv2d_30_conv2d_readvariableop_resource*'
_output_shapes
:H?*
dtype02F
Ddenoising_autoencoder_4/sequential_8/conv2d_30/Conv2D/ReadVariableOp?
5denoising_autoencoder_4/sequential_8/conv2d_30/Conv2DConv2DAdenoising_autoencoder_4/sequential_8/conv2d_29/Relu:activations:0Ldenoising_autoencoder_4/sequential_8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
27
5denoising_autoencoder_4/sequential_8/conv2d_30/Conv2D?
Edenoising_autoencoder_4/sequential_8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOpNdenoising_autoencoder_4_sequential_8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Edenoising_autoencoder_4/sequential_8/conv2d_30/BiasAdd/ReadVariableOp?
6denoising_autoencoder_4/sequential_8/conv2d_30/BiasAddBiasAdd>denoising_autoencoder_4/sequential_8/conv2d_30/Conv2D:output:0Mdenoising_autoencoder_4/sequential_8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????28
6denoising_autoencoder_4/sequential_8/conv2d_30/BiasAdd?
3denoising_autoencoder_4/sequential_8/conv2d_30/ReluRelu?denoising_autoencoder_4/sequential_8/conv2d_30/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????25
3denoising_autoencoder_4/sequential_8/conv2d_30/Relu?
Idenoising_autoencoder_4/sequential_8/batch_normalization_8/ReadVariableOpReadVariableOpRdenoising_autoencoder_4_sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02K
Idenoising_autoencoder_4/sequential_8/batch_normalization_8/ReadVariableOp?
Kdenoising_autoencoder_4/sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOpTdenoising_autoencoder_4_sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02M
Kdenoising_autoencoder_4/sequential_8/batch_normalization_8/ReadVariableOp_1?
Zdenoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpcdenoising_autoencoder_4_sequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02\
Zdenoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
\denoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpedenoising_autoencoder_4_sequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02^
\denoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
Kdenoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3Adenoising_autoencoder_4/sequential_8/conv2d_30/Relu:activations:0Qdenoising_autoencoder_4/sequential_8/batch_normalization_8/ReadVariableOp:value:0Sdenoising_autoencoder_4/sequential_8/batch_normalization_8/ReadVariableOp_1:value:0bdenoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0ddenoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2M
Kdenoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3?
<denoising_autoencoder_4/sequential_8/max_pooling2d_4/MaxPoolMaxPoolOdenoising_autoencoder_4/sequential_8/batch_normalization_8/FusedBatchNormV3:y:0*2
_output_shapes 
:????????????*
ksize
*
paddingSAME*
strides
2>
<denoising_autoencoder_4/sequential_8/max_pooling2d_4/MaxPool?
7denoising_autoencoder_4/sequential_8/dropout_4/IdentityIdentityEdenoising_autoencoder_4/sequential_8/max_pooling2d_4/MaxPool:output:0*
T0*2
_output_shapes 
:????????????29
7denoising_autoencoder_4/sequential_8/dropout_4/Identity?
Ddenoising_autoencoder_4/sequential_9/conv2d_31/Conv2D/ReadVariableOpReadVariableOpMdenoising_autoencoder_4_sequential_9_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02F
Ddenoising_autoencoder_4/sequential_9/conv2d_31/Conv2D/ReadVariableOp?
5denoising_autoencoder_4/sequential_9/conv2d_31/Conv2DConv2D@denoising_autoencoder_4/sequential_8/dropout_4/Identity:output:0Ldenoising_autoencoder_4/sequential_9/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
27
5denoising_autoencoder_4/sequential_9/conv2d_31/Conv2D?
Edenoising_autoencoder_4/sequential_9/conv2d_31/BiasAdd/ReadVariableOpReadVariableOpNdenoising_autoencoder_4_sequential_9_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Edenoising_autoencoder_4/sequential_9/conv2d_31/BiasAdd/ReadVariableOp?
6denoising_autoencoder_4/sequential_9/conv2d_31/BiasAddBiasAdd>denoising_autoencoder_4/sequential_9/conv2d_31/Conv2D:output:0Mdenoising_autoencoder_4/sequential_9/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????28
6denoising_autoencoder_4/sequential_9/conv2d_31/BiasAdd?
3denoising_autoencoder_4/sequential_9/conv2d_31/ReluRelu?denoising_autoencoder_4/sequential_9/conv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????25
3denoising_autoencoder_4/sequential_9/conv2d_31/Relu?
Ddenoising_autoencoder_4/sequential_9/conv2d_32/Conv2D/ReadVariableOpReadVariableOpMdenoising_autoencoder_4_sequential_9_conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02F
Ddenoising_autoencoder_4/sequential_9/conv2d_32/Conv2D/ReadVariableOp?
5denoising_autoencoder_4/sequential_9/conv2d_32/Conv2DConv2DAdenoising_autoencoder_4/sequential_9/conv2d_31/Relu:activations:0Ldenoising_autoencoder_4/sequential_9/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
27
5denoising_autoencoder_4/sequential_9/conv2d_32/Conv2D?
Edenoising_autoencoder_4/sequential_9/conv2d_32/BiasAdd/ReadVariableOpReadVariableOpNdenoising_autoencoder_4_sequential_9_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02G
Edenoising_autoencoder_4/sequential_9/conv2d_32/BiasAdd/ReadVariableOp?
6denoising_autoencoder_4/sequential_9/conv2d_32/BiasAddBiasAdd>denoising_autoencoder_4/sequential_9/conv2d_32/Conv2D:output:0Mdenoising_autoencoder_4/sequential_9/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H28
6denoising_autoencoder_4/sequential_9/conv2d_32/BiasAdd?
3denoising_autoencoder_4/sequential_9/conv2d_32/ReluRelu?denoising_autoencoder_4/sequential_9/conv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H25
3denoising_autoencoder_4/sequential_9/conv2d_32/Relu?
Ddenoising_autoencoder_4/sequential_9/conv2d_33/Conv2D/ReadVariableOpReadVariableOpMdenoising_autoencoder_4_sequential_9_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02F
Ddenoising_autoencoder_4/sequential_9/conv2d_33/Conv2D/ReadVariableOp?
5denoising_autoencoder_4/sequential_9/conv2d_33/Conv2DConv2DAdenoising_autoencoder_4/sequential_9/conv2d_32/Relu:activations:0Ldenoising_autoencoder_4/sequential_9/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
27
5denoising_autoencoder_4/sequential_9/conv2d_33/Conv2D?
Edenoising_autoencoder_4/sequential_9/conv2d_33/BiasAdd/ReadVariableOpReadVariableOpNdenoising_autoencoder_4_sequential_9_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02G
Edenoising_autoencoder_4/sequential_9/conv2d_33/BiasAdd/ReadVariableOp?
6denoising_autoencoder_4/sequential_9/conv2d_33/BiasAddBiasAdd>denoising_autoencoder_4/sequential_9/conv2d_33/Conv2D:output:0Mdenoising_autoencoder_4/sequential_9/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????028
6denoising_autoencoder_4/sequential_9/conv2d_33/BiasAdd?
3denoising_autoencoder_4/sequential_9/conv2d_33/ReluRelu?denoising_autoencoder_4/sequential_9/conv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????025
3denoising_autoencoder_4/sequential_9/conv2d_33/Relu?
Idenoising_autoencoder_4/sequential_9/batch_normalization_9/ReadVariableOpReadVariableOpRdenoising_autoencoder_4_sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype02K
Idenoising_autoencoder_4/sequential_9/batch_normalization_9/ReadVariableOp?
Kdenoising_autoencoder_4/sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOpTdenoising_autoencoder_4_sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype02M
Kdenoising_autoencoder_4/sequential_9/batch_normalization_9/ReadVariableOp_1?
Zdenoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpcdenoising_autoencoder_4_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02\
Zdenoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
\denoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpedenoising_autoencoder_4_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02^
\denoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
Kdenoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3Adenoising_autoencoder_4/sequential_9/conv2d_33/Relu:activations:0Qdenoising_autoencoder_4/sequential_9/batch_normalization_9/ReadVariableOp:value:0Sdenoising_autoencoder_4/sequential_9/batch_normalization_9/ReadVariableOp_1:value:0bdenoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0ddenoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2M
Kdenoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3?
:denoising_autoencoder_4/sequential_9/up_sampling2d_4/ShapeShapeOdenoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2<
:denoising_autoencoder_4/sequential_9/up_sampling2d_4/Shape?
Hdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
Hdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stack?
Jdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stack_1?
Jdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stack_2?
Bdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_sliceStridedSliceCdenoising_autoencoder_4/sequential_9/up_sampling2d_4/Shape:output:0Qdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stack:output:0Sdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stack_1:output:0Sdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2D
Bdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice?
:denoising_autoencoder_4/sequential_9/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2<
:denoising_autoencoder_4/sequential_9/up_sampling2d_4/Const?
8denoising_autoencoder_4/sequential_9/up_sampling2d_4/mulMulKdenoising_autoencoder_4/sequential_9/up_sampling2d_4/strided_slice:output:0Cdenoising_autoencoder_4/sequential_9/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2:
8denoising_autoencoder_4/sequential_9/up_sampling2d_4/mul?
Qdenoising_autoencoder_4/sequential_9/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborOdenoising_autoencoder_4/sequential_9/batch_normalization_9/FusedBatchNormV3:y:0<denoising_autoencoder_4/sequential_9/up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2S
Qdenoising_autoencoder_4/sequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor?
Ddenoising_autoencoder_4/sequential_9/conv2d_34/Conv2D/ReadVariableOpReadVariableOpMdenoising_autoencoder_4_sequential_9_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02F
Ddenoising_autoencoder_4/sequential_9/conv2d_34/Conv2D/ReadVariableOp?
5denoising_autoencoder_4/sequential_9/conv2d_34/Conv2DConv2Dbdenoising_autoencoder_4/sequential_9/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0Ldenoising_autoencoder_4/sequential_9/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
27
5denoising_autoencoder_4/sequential_9/conv2d_34/Conv2D?
Edenoising_autoencoder_4/sequential_9/conv2d_34/BiasAdd/ReadVariableOpReadVariableOpNdenoising_autoencoder_4_sequential_9_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Edenoising_autoencoder_4/sequential_9/conv2d_34/BiasAdd/ReadVariableOp?
6denoising_autoencoder_4/sequential_9/conv2d_34/BiasAddBiasAdd>denoising_autoencoder_4/sequential_9/conv2d_34/Conv2D:output:0Mdenoising_autoencoder_4/sequential_9/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????28
6denoising_autoencoder_4/sequential_9/conv2d_34/BiasAdd?
6denoising_autoencoder_4/sequential_9/conv2d_34/SigmoidSigmoid?denoising_autoencoder_4/sequential_9/conv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????28
6denoising_autoencoder_4/sequential_9/conv2d_34/Sigmoid?
IdentityIdentity:denoising_autoencoder_4/sequential_9/conv2d_34/Sigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????:::::::::::::::::::::::Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?B
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_34847

inputs,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource,
(conv2d_32_conv2d_readvariableop_resource-
)conv2d_32_biasadd_readvariableop_resource,
(conv2d_33_conv2d_readvariableop_resource-
)conv2d_33_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource
identity??$batch_normalization_9/AssignNewValue?&batch_normalization_9/AssignNewValue_1?
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_31/Conv2D/ReadVariableOp?
conv2d_31/Conv2DConv2Dinputs'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_31/Conv2D?
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp?
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_31/BiasAdd?
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
conv2d_31/Relu?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*'
_output_shapes
:?H*
dtype02!
conv2d_32/Conv2D/ReadVariableOp?
conv2d_32/Conv2DConv2Dconv2d_31/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H*
paddingSAME*
strides
2
conv2d_32/Conv2D?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????H2
conv2d_32/BiasAdd?
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:???????????H2
conv2d_32/Relu?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:H0*
dtype02!
conv2d_33/Conv2D/ReadVariableOp?
conv2d_33/Conv2DConv2Dconv2d_32/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
2
conv2d_33/Conv2D?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????02
conv2d_33/BiasAdd?
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:???????????02
conv2d_33/Relu?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_33/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_9/FusedBatchNormV3?
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue?
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1?
up_sampling2d_4/ShapeShape*batch_normalization_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape?
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack?
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1?
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2?
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor*batch_normalization_9/FusedBatchNormV3:y:0up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:???????????0*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_34/BiasAdd?
conv2d_34/SigmoidSigmoidconv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_34/Sigmoid?
IdentityIdentityconv2d_34/Sigmoid:y:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:????????????::::::::::::2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35207

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
7__inference_denoising_autoencoder_4_layer_call_fn_34650
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_339132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:???????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
5__inference_batch_normalization_9_layer_call_fn_35453

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_332422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
,__inference_sequential_8_layer_call_fn_33180
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_331572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:???????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?
f
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_33297

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????F
output_1:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
encoder
decoder
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"class_name": "DenoisingAutoencoder", "name": "denoising_autoencoder_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DenoisingAutoencoder"}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_absolute_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?9
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer-5
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?7
_tf_keras_sequential?6{"class_name": "Sequential", "name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 420, 540, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 420, 540, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 420, 540, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}}}
?A
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?>
_tf_keras_sequential?>{"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 210, 270, 144]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_31_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 210, 270, 144]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 210, 270, 144]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_31_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?
iter

beta_1

beta_2
	 decay
!learning_rate"m?#m?$m?%m?&m?'m?(m?)m?,m?-m?.m?/m?0m?1m?2m?3m?6m?7m?"v?#v?$v?%v?&v?'v?(v?)v?,v?-v?.v?/v?0v?1v?2v?3v?6v?7v?"
	optimizer
 "
trackable_list_wrapper
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721"
trackable_list_wrapper
?
"0
#1
$2
%3
&4
'5
(6
)7
,8
-9
.10
/11
012
113
214
315
616
717"
trackable_list_wrapper
?
regularization_losses
8layer_regularization_losses

9layers
:non_trainable_variables
;metrics
	variables
trainable_variables
<layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?	

"kernel
#bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 420, 540, 1]}}
?	

$kernel
%bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 420, 540, 48]}}
?	

&kernel
'bias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 72}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 420, 540, 72]}}
?	
Iaxis
	(gamma
)beta
*moving_mean
+moving_variance
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 420, 540, 144]}}
?
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
f
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9"
trackable_list_wrapper
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
?
regularization_losses
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
Ymetrics
	variables
trainable_variables
Zlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

[_inbound_nodes

,kernel
-bias
\_outbound_nodes
]regularization_losses
^	variables
_trainable_variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [12, 210, 270, 144]}}
?

a_inbound_nodes

.kernel
/bias
b_outbound_nodes
cregularization_losses
d	variables
etrainable_variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [12, 210, 270, 144]}}
?

g_inbound_nodes

0kernel
1bias
h_outbound_nodes
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 72}}}, "build_input_shape": {"class_name": "TensorShape", "items": [12, 210, 270, 72]}}
?	
m_inbound_nodes
naxis
	2gamma
3beta
4moving_mean
5moving_variance
o_outbound_nodes
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [12, 210, 270, 48]}}
?
t_inbound_nodes
u_outbound_nodes
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

z_inbound_nodes

6kernel
7bias
{regularization_losses
|	variables
}trainable_variables
~	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [12, 420, 540, 48]}}
 "
trackable_list_wrapper
v
,0
-1
.2
/3
04
15
26
37
48
59
610
711"
trackable_list_wrapper
f
,0
-1
.2
/3
04
15
26
37
68
79"
trackable_list_wrapper
?
regularization_losses
layer_regularization_losses
?layers
?non_trainable_variables
?metrics
	variables
trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(02conv2d_28/kernel
:02conv2d_28/bias
*:(0H2conv2d_29/kernel
:H2conv2d_29/bias
+:)H?2conv2d_30/kernel
:?2conv2d_30/bias
*:(?2batch_normalization_8/gamma
):'?2batch_normalization_8/beta
2:0? (2!batch_normalization_8/moving_mean
6:4? (2%batch_normalization_8/moving_variance
,:*??2conv2d_31/kernel
:?2conv2d_31/bias
+:)?H2conv2d_32/kernel
:H2conv2d_32/bias
*:(H02conv2d_33/kernel
:02conv2d_33/bias
):'02batch_normalization_9/gamma
(:&02batch_normalization_9/beta
1:/0 (2!batch_normalization_9/moving_mean
5:30 (2%batch_normalization_9/moving_variance
*:(02conv2d_34/kernel
:2conv2d_34/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
*0
+1
42
53"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
=regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
>	variables
?trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
Aregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
B	variables
Ctrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
Eregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
F	variables
Gtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
(0
)1
*2
+3"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
Jregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
K	variables
Ltrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
O	variables
Ptrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
S	variables
Ttrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
]regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
^	variables
_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
cregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
d	variables
etrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
iregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
j	variables
ktrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
20
31
42
53"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
pregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
q	variables
rtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
w	variables
xtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
{regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
|	variables
}trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mean_absolute_error", "dtype": "float32", "config": {"name": "mean_absolute_error", "dtype": "float32", "fn": "mean_absolute_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-02Adam/conv2d_28/kernel/m
!:02Adam/conv2d_28/bias/m
/:-0H2Adam/conv2d_29/kernel/m
!:H2Adam/conv2d_29/bias/m
0:.H?2Adam/conv2d_30/kernel/m
": ?2Adam/conv2d_30/bias/m
/:-?2"Adam/batch_normalization_8/gamma/m
.:,?2!Adam/batch_normalization_8/beta/m
1:/??2Adam/conv2d_31/kernel/m
": ?2Adam/conv2d_31/bias/m
0:.?H2Adam/conv2d_32/kernel/m
!:H2Adam/conv2d_32/bias/m
/:-H02Adam/conv2d_33/kernel/m
!:02Adam/conv2d_33/bias/m
.:,02"Adam/batch_normalization_9/gamma/m
-:+02!Adam/batch_normalization_9/beta/m
/:-02Adam/conv2d_34/kernel/m
!:2Adam/conv2d_34/bias/m
/:-02Adam/conv2d_28/kernel/v
!:02Adam/conv2d_28/bias/v
/:-0H2Adam/conv2d_29/kernel/v
!:H2Adam/conv2d_29/bias/v
0:.H?2Adam/conv2d_30/kernel/v
": ?2Adam/conv2d_30/bias/v
/:-?2"Adam/batch_normalization_8/gamma/v
.:,?2!Adam/batch_normalization_8/beta/v
1:/??2Adam/conv2d_31/kernel/v
": ?2Adam/conv2d_31/bias/v
0:.?H2Adam/conv2d_32/kernel/v
!:H2Adam/conv2d_32/bias/v
/:-H02Adam/conv2d_33/kernel/v
!:02Adam/conv2d_33/bias/v
.:,02"Adam/batch_normalization_9/gamma/v
-:+02!Adam/batch_normalization_9/beta/v
/:-02Adam/conv2d_34/kernel/v
!:2Adam/conv2d_34/bias/v
?2?
 __inference__wrapped_model_32733?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_1???????????
?2?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34552
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34461
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34170
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34261?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_denoising_autoencoder_4_layer_call_fn_34601
7__inference_denoising_autoencoder_4_layer_call_fn_34359
7__inference_denoising_autoencoder_4_layer_call_fn_34310
7__inference_denoising_autoencoder_4_layer_call_fn_34650?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_sequential_8_layer_call_and_return_conditional_losses_34741
G__inference_sequential_8_layer_call_and_return_conditional_losses_34700
G__inference_sequential_8_layer_call_and_return_conditional_losses_33069
G__inference_sequential_8_layer_call_and_return_conditional_losses_33039?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_8_layer_call_fn_34766
,__inference_sequential_8_layer_call_fn_33180
,__inference_sequential_8_layer_call_fn_33125
,__inference_sequential_8_layer_call_fn_34791?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_9_layer_call_and_return_conditional_losses_34901
G__inference_sequential_9_layer_call_and_return_conditional_losses_35069
G__inference_sequential_9_layer_call_and_return_conditional_losses_35015
G__inference_sequential_9_layer_call_and_return_conditional_losses_34847?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_9_layer_call_fn_35098
,__inference_sequential_9_layer_call_fn_34930
,__inference_sequential_9_layer_call_fn_35127
,__inference_sequential_9_layer_call_fn_34959?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
2B0
#__inference_signature_wrapper_34068input_1
?2?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_35138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_28_layer_call_fn_35147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_35158?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_29_layer_call_fn_35167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_30_layer_call_and_return_conditional_losses_35178?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_30_layer_call_fn_35187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35225
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35207
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35271
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35289?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_8_layer_call_fn_35302
5__inference_batch_normalization_8_layer_call_fn_35238
5__inference_batch_normalization_8_layer_call_fn_35251
5__inference_batch_normalization_8_layer_call_fn_35315?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_32843?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_4_layer_call_fn_32849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_dropout_4_layer_call_and_return_conditional_losses_35332
D__inference_dropout_4_layer_call_and_return_conditional_losses_35327?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_4_layer_call_fn_35342
)__inference_dropout_4_layer_call_fn_35337?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_31_layer_call_and_return_conditional_losses_35353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_31_layer_call_fn_35362?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_32_layer_call_and_return_conditional_losses_35373?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_32_layer_call_fn_35382?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_33_layer_call_and_return_conditional_losses_35393?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_33_layer_call_fn_35402?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35504
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35440
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35422
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35486?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_9_layer_call_fn_35517
5__inference_batch_normalization_9_layer_call_fn_35453
5__inference_batch_normalization_9_layer_call_fn_35530
5__inference_batch_normalization_9_layer_call_fn_35466?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_33297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_up_sampling2d_4_layer_call_fn_33303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_conv2d_34_layer_call_and_return_conditional_losses_35541?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_34_layer_call_fn_35550?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_32733?"#$%&'()*+,-./01234567:?7
0?-
+?(
input_1???????????
? "=?:
8
output_1,?)
output_1????????????
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35207x()*+>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35225x()*+>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35271?()*+N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_35289?()*+N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_batch_normalization_8_layer_call_fn_35238k()*+>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
5__inference_batch_normalization_8_layer_call_fn_35251k()*+>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
5__inference_batch_normalization_8_layer_call_fn_35302?()*+N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
5__inference_batch_normalization_8_layer_call_fn_35315?()*+N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35422?2345M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35440?2345M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35486v2345=?:
3?0
*?'
inputs???????????0
p
? "/?,
%?"
0???????????0
? ?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_35504v2345=?:
3?0
*?'
inputs???????????0
p 
? "/?,
%?"
0???????????0
? ?
5__inference_batch_normalization_9_layer_call_fn_35453?2345M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
5__inference_batch_normalization_9_layer_call_fn_35466?2345M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
5__inference_batch_normalization_9_layer_call_fn_35517i2345=?:
3?0
*?'
inputs???????????0
p
? ""????????????0?
5__inference_batch_normalization_9_layer_call_fn_35530i2345=?:
3?0
*?'
inputs???????????0
p 
? ""????????????0?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_35138p"#9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????0
? ?
)__inference_conv2d_28_layer_call_fn_35147c"#9?6
/?,
*?'
inputs???????????
? ""????????????0?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_35158p$%9?6
/?,
*?'
inputs???????????0
? "/?,
%?"
0???????????H
? ?
)__inference_conv2d_29_layer_call_fn_35167c$%9?6
/?,
*?'
inputs???????????0
? ""????????????H?
D__inference_conv2d_30_layer_call_and_return_conditional_losses_35178q&'9?6
/?,
*?'
inputs???????????H
? "0?-
&?#
0????????????
? ?
)__inference_conv2d_30_layer_call_fn_35187d&'9?6
/?,
*?'
inputs???????????H
? "#? ?????????????
D__inference_conv2d_31_layer_call_and_return_conditional_losses_35353r,-:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
)__inference_conv2d_31_layer_call_fn_35362e,-:?7
0?-
+?(
inputs????????????
? "#? ?????????????
D__inference_conv2d_32_layer_call_and_return_conditional_losses_35373q./:?7
0?-
+?(
inputs????????????
? "/?,
%?"
0???????????H
? ?
)__inference_conv2d_32_layer_call_fn_35382d./:?7
0?-
+?(
inputs????????????
? ""????????????H?
D__inference_conv2d_33_layer_call_and_return_conditional_losses_35393p019?6
/?,
*?'
inputs???????????H
? "/?,
%?"
0???????????0
? ?
)__inference_conv2d_33_layer_call_fn_35402c019?6
/?,
*?'
inputs???????????H
? ""????????????0?
D__inference_conv2d_34_layer_call_and_return_conditional_losses_35541?67I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv2d_34_layer_call_fn_35550?67I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+????????????????????????????
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34170?"#$%&'()*+,-./01234567>?;
4?1
+?(
input_1???????????
p
? "/?,
%?"
0???????????
? ?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34261?"#$%&'()*+,-./01234567>?;
4?1
+?(
input_1???????????
p 
? "/?,
%?"
0???????????
? ?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34461?"#$%&'()*+,-./012345678?5
.?+
%?"
x???????????
p
? "/?,
%?"
0???????????
? ?
R__inference_denoising_autoencoder_4_layer_call_and_return_conditional_losses_34552?"#$%&'()*+,-./012345678?5
.?+
%?"
x???????????
p 
? "/?,
%?"
0???????????
? ?
7__inference_denoising_autoencoder_4_layer_call_fn_34310?"#$%&'()*+,-./01234567>?;
4?1
+?(
input_1???????????
p
? "2?/+????????????????????????????
7__inference_denoising_autoencoder_4_layer_call_fn_34359?"#$%&'()*+,-./01234567>?;
4?1
+?(
input_1???????????
p 
? "2?/+????????????????????????????
7__inference_denoising_autoencoder_4_layer_call_fn_34601?"#$%&'()*+,-./012345678?5
.?+
%?"
x???????????
p
? "2?/+????????????????????????????
7__inference_denoising_autoencoder_4_layer_call_fn_34650?"#$%&'()*+,-./012345678?5
.?+
%?"
x???????????
p 
? "2?/+????????????????????????????
D__inference_dropout_4_layer_call_and_return_conditional_losses_35327r>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_35332r>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
)__inference_dropout_4_layer_call_fn_35337e>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
)__inference_dropout_4_layer_call_fn_35342e>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_32843?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_4_layer_call_fn_32849?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_sequential_8_layer_call_and_return_conditional_losses_33039?
"#$%&'()*+B??
8?5
+?(
input_5???????????
p

 
? "0?-
&?#
0????????????
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_33069?
"#$%&'()*+B??
8?5
+?(
input_5???????????
p 

 
? "0?-
&?#
0????????????
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_34700?
"#$%&'()*+A?>
7?4
*?'
inputs???????????
p

 
? "0?-
&?#
0????????????
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_34741?
"#$%&'()*+A?>
7?4
*?'
inputs???????????
p 

 
? "0?-
&?#
0????????????
? ?
,__inference_sequential_8_layer_call_fn_33125u
"#$%&'()*+B??
8?5
+?(
input_5???????????
p

 
? "#? ?????????????
,__inference_sequential_8_layer_call_fn_33180u
"#$%&'()*+B??
8?5
+?(
input_5???????????
p 

 
? "#? ?????????????
,__inference_sequential_8_layer_call_fn_34766t
"#$%&'()*+A?>
7?4
*?'
inputs???????????
p

 
? "#? ?????????????
,__inference_sequential_8_layer_call_fn_34791t
"#$%&'()*+A?>
7?4
*?'
inputs???????????
p 

 
? "#? ?????????????
G__inference_sequential_9_layer_call_and_return_conditional_losses_34847?,-./01234567B??
8?5
+?(
inputs????????????
p

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_34901?,-./01234567B??
8?5
+?(
inputs????????????
p 

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_35015?,-./01234567K?H
A?>
4?1
conv2d_31_input????????????
p

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_35069?,-./01234567K?H
A?>
4?1
conv2d_31_input????????????
p 

 
? "/?,
%?"
0???????????
? ?
,__inference_sequential_9_layer_call_fn_34930?,-./01234567B??
8?5
+?(
inputs????????????
p

 
? "2?/+????????????????????????????
,__inference_sequential_9_layer_call_fn_34959?,-./01234567B??
8?5
+?(
inputs????????????
p 

 
? "2?/+????????????????????????????
,__inference_sequential_9_layer_call_fn_35098?,-./01234567K?H
A?>
4?1
conv2d_31_input????????????
p

 
? "2?/+????????????????????????????
,__inference_sequential_9_layer_call_fn_35127?,-./01234567K?H
A?>
4?1
conv2d_31_input????????????
p 

 
? "2?/+????????????????????????????
#__inference_signature_wrapper_34068?"#$%&'()*+,-./01234567E?B
? 
;?8
6
input_1+?(
input_1???????????"=?:
8
output_1,?)
output_1????????????
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_33297?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_4_layer_call_fn_33303?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????
½	
â
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718À
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
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

Nadam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameNadam/dense/kernel/m
}
(Nadam/dense/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/m*
_output_shapes

:*
dtype0
|
Nadam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameNadam/dense/bias/m
u
&Nadam/dense/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense/bias/m*
_output_shapes
:*
dtype0

Nadam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/dense_1/kernel/m

*Nadam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/m*
_output_shapes

:*
dtype0

Nadam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_1/bias/m
y
(Nadam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/m*
_output_shapes
:*
dtype0

Nadam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameNadam/dense/kernel/v
}
(Nadam/dense/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/v*
_output_shapes

:*
dtype0
|
Nadam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameNadam/dense/bias/v
u
&Nadam/dense/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense/bias/v*
_output_shapes
:*
dtype0

Nadam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/dense_1/kernel/v

*Nadam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/v*
_output_shapes

:*
dtype0

Nadam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_1/bias/v
y
(Nadam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
´
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ï
valueåBâ BÛ
Ì
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
h


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api

iter

beta_1

beta_2
	decay
learning_rate
momentum_cache
m9m:m;m<
v=v>v?v@
 


0
1
2
3


0
1
2
3
­
 layer_regularization_losses

!layers
regularization_losses
"layer_metrics
	variables
#non_trainable_variables
$metrics
trainable_variables
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
­
trainable_variables
%layer_regularization_losses

&layers
regularization_losses
'layer_metrics
	variables
(metrics
)non_trainable_variables
 
 
 
­
trainable_variables
*layer_regularization_losses

+layers
regularization_losses
,layer_metrics
	variables
-metrics
.non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
/layer_regularization_losses

0layers
regularization_losses
1layer_metrics
	variables
2metrics
3non_trainable_variables
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 
 

40
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
4
	5total
	6count
7	variables
8	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

50
61

7	variables
|z
VARIABLE_VALUENadam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUENadam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUENadam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_dense_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ö
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_402503
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Nadam/dense/kernel/m/Read/ReadVariableOp&Nadam/dense/bias/m/Read/ReadVariableOp*Nadam/dense_1/kernel/m/Read/ReadVariableOp(Nadam/dense_1/bias/m/Read/ReadVariableOp(Nadam/dense/kernel/v/Read/ReadVariableOp&Nadam/dense/bias/v/Read/ReadVariableOp*Nadam/dense_1/kernel/v/Read/ReadVariableOp(Nadam/dense_1/bias/v/Read/ReadVariableOpConst*!
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_403165

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/dense/kernel/mNadam/dense/bias/mNadam/dense_1/kernel/mNadam/dense_1/bias/mNadam/dense/kernel/vNadam/dense/bias/vNadam/dense_1/kernel/vNadam/dense_1/bias/v* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_403235×­
©
b
C__inference_dropout_layer_call_and_return_conditional_losses_403025

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


&__inference_dense_layer_call_fn_402950

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4022762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
^
Á
F__inference_sequential_layer_call_and_return_conditional_losses_402648

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd_
dense/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/Const
dense/LessEqual	LessEqualdense/BiasAdd:output:0dense/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/LessEqualc
dense/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/Const_1
dense/GreaterGreaterdense/BiasAdd:output:0dense/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Greaterc
dense/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense/Const_2
dense/Greater_1Greaterdense/BiasAdd:output:0dense/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Greater_1c
dense/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense/Const_3
dense/LessEqual_1	LessEqualdense/BiasAdd:output:0dense/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/LessEqual_1_
dense/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/sub/y}
	dense/subSubdense/BiasAdd:output:0dense/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/subi
dense/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2/e
dense/SelectV2SelectV2dense/LessEqual:z:0dense/sub:z:0dense/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2h
	dense/ExpExpdense/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/Exp_
dense/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
dense/mul/xt
	dense/mulMuldense/mul/x:output:0dense/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/mulc
dense/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
dense/mul_1/x
dense/mul_1Muldense/mul_1/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_1c
dense/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
dense/sub_1/y|
dense/sub_1Subdense/mul_1:z:0dense/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/sub_1g
dense/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/truediv/x
dense/truedivRealDivdense/truediv/x:output:0dense/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/truedivc
dense/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/sub_2/x~
dense/sub_2Subdense/sub_2/x:output:0dense/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/sub_2c
dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
dense/mul_2/x
dense/mul_2Muldense/mul_2/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_2_
dense/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/add/yx
	dense/addAddV2dense/mul_2:z:0dense/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/addm
dense/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2_1/e¥
dense/SelectV2_1SelectV2dense/LessEqual_1:z:0dense/add:z:0dense/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_1m
dense/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2_2/e­
dense/SelectV2_2SelectV2dense/Greater:z:0dense/SelectV2_1:output:0dense/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_2j
	dense/LogLogdense/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/Logc
dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
dense/mul_3/xz
dense/mul_3Muldense/mul_3/x:output:0dense/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_3c
dense/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/add_1/y~
dense/add_1AddV2dense/mul_3:z:0dense/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/add_1
dense/SelectV2_3SelectV2dense/Greater_1:z:0dense/sub_2:z:0dense/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_3¡
dense/SelectV2_4SelectV2dense/LessEqual:z:0dense/mul:z:0dense/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_4}
dropout/IdentityIdentitydense/SelectV2_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Identity¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddc
dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/Const
dense_1/LessEqual	LessEqualdense_1/BiasAdd:output:0dense_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/LessEqualg
dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/Const_1
dense_1/GreaterGreaterdense_1/BiasAdd:output:0dense_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Greaterg
dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense_1/Const_2
dense_1/Greater_1Greaterdense_1/BiasAdd:output:0dense_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Greater_1g
dense_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense_1/Const_3
dense_1/LessEqual_1	LessEqualdense_1/BiasAdd:output:0dense_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/LessEqual_1c
dense_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/sub/y
dense_1/subSubdense_1/BiasAdd:output:0dense_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/subm
dense_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2/e§
dense_1/SelectV2SelectV2dense_1/LessEqual:z:0dense_1/sub:z:0dense_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2n
dense_1/ExpExpdense_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Expc
dense_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
dense_1/mul/x|
dense_1/mulMuldense_1/mul/x:output:0dense_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mulg
dense_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
dense_1/mul_1/x
dense_1/mul_1Muldense_1/mul_1/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_1g
dense_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
dense_1/sub_1/y
dense_1/sub_1Subdense_1/mul_1:z:0dense_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/sub_1k
dense_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/truediv/x
dense_1/truedivRealDivdense_1/truediv/x:output:0dense_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/truedivg
dense_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/sub_2/x
dense_1/sub_2Subdense_1/sub_2/x:output:0dense_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/sub_2g
dense_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
dense_1/mul_2/x
dense_1/mul_2Muldense_1/mul_2/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_2c
dense_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/add/y
dense_1/addAddV2dense_1/mul_2:z:0dense_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/addq
dense_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2_1/e¯
dense_1/SelectV2_1SelectV2dense_1/LessEqual_1:z:0dense_1/add:z:0dense_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_1q
dense_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2_2/e·
dense_1/SelectV2_2SelectV2dense_1/Greater:z:0dense_1/SelectV2_1:output:0dense_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_2p
dense_1/LogLogdense_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Logg
dense_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
dense_1/mul_3/x
dense_1/mul_3Muldense_1/mul_3/x:output:0dense_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_3g
dense_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/add_1/y
dense_1/add_1AddV2dense_1/mul_3:z:0dense_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/add_1£
dense_1/SelectV2_3SelectV2dense_1/Greater_1:z:0dense_1/sub_2:z:0dense_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_3«
dense_1/SelectV2_4SelectV2dense_1/LessEqual:z:0dense_1/mul:z:0dense_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_4í
IdentityIdentitydense_1/SelectV2_4:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ó
+__inference_sequential_layer_call_fn_402555
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4024282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
­^
Æ
F__inference_sequential_layer_call_and_return_conditional_losses_402841
dense_input6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense_input#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd_
dense/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/Const
dense/LessEqual	LessEqualdense/BiasAdd:output:0dense/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/LessEqualc
dense/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/Const_1
dense/GreaterGreaterdense/BiasAdd:output:0dense/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Greaterc
dense/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense/Const_2
dense/Greater_1Greaterdense/BiasAdd:output:0dense/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Greater_1c
dense/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense/Const_3
dense/LessEqual_1	LessEqualdense/BiasAdd:output:0dense/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/LessEqual_1_
dense/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/sub/y}
	dense/subSubdense/BiasAdd:output:0dense/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/subi
dense/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2/e
dense/SelectV2SelectV2dense/LessEqual:z:0dense/sub:z:0dense/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2h
	dense/ExpExpdense/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/Exp_
dense/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
dense/mul/xt
	dense/mulMuldense/mul/x:output:0dense/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/mulc
dense/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
dense/mul_1/x
dense/mul_1Muldense/mul_1/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_1c
dense/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
dense/sub_1/y|
dense/sub_1Subdense/mul_1:z:0dense/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/sub_1g
dense/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/truediv/x
dense/truedivRealDivdense/truediv/x:output:0dense/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/truedivc
dense/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/sub_2/x~
dense/sub_2Subdense/sub_2/x:output:0dense/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/sub_2c
dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
dense/mul_2/x
dense/mul_2Muldense/mul_2/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_2_
dense/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/add/yx
	dense/addAddV2dense/mul_2:z:0dense/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/addm
dense/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2_1/e¥
dense/SelectV2_1SelectV2dense/LessEqual_1:z:0dense/add:z:0dense/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_1m
dense/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2_2/e­
dense/SelectV2_2SelectV2dense/Greater:z:0dense/SelectV2_1:output:0dense/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_2j
	dense/LogLogdense/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/Logc
dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
dense/mul_3/xz
dense/mul_3Muldense/mul_3/x:output:0dense/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_3c
dense/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/add_1/y~
dense/add_1AddV2dense/mul_3:z:0dense/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/add_1
dense/SelectV2_3SelectV2dense/Greater_1:z:0dense/sub_2:z:0dense/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_3¡
dense/SelectV2_4SelectV2dense/LessEqual:z:0dense/mul:z:0dense/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_4}
dropout/IdentityIdentitydense/SelectV2_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Identity¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddc
dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/Const
dense_1/LessEqual	LessEqualdense_1/BiasAdd:output:0dense_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/LessEqualg
dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/Const_1
dense_1/GreaterGreaterdense_1/BiasAdd:output:0dense_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Greaterg
dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense_1/Const_2
dense_1/Greater_1Greaterdense_1/BiasAdd:output:0dense_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Greater_1g
dense_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense_1/Const_3
dense_1/LessEqual_1	LessEqualdense_1/BiasAdd:output:0dense_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/LessEqual_1c
dense_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/sub/y
dense_1/subSubdense_1/BiasAdd:output:0dense_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/subm
dense_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2/e§
dense_1/SelectV2SelectV2dense_1/LessEqual:z:0dense_1/sub:z:0dense_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2n
dense_1/ExpExpdense_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Expc
dense_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
dense_1/mul/x|
dense_1/mulMuldense_1/mul/x:output:0dense_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mulg
dense_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
dense_1/mul_1/x
dense_1/mul_1Muldense_1/mul_1/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_1g
dense_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
dense_1/sub_1/y
dense_1/sub_1Subdense_1/mul_1:z:0dense_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/sub_1k
dense_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/truediv/x
dense_1/truedivRealDivdense_1/truediv/x:output:0dense_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/truedivg
dense_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/sub_2/x
dense_1/sub_2Subdense_1/sub_2/x:output:0dense_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/sub_2g
dense_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
dense_1/mul_2/x
dense_1/mul_2Muldense_1/mul_2/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_2c
dense_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/add/y
dense_1/addAddV2dense_1/mul_2:z:0dense_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/addq
dense_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2_1/e¯
dense_1/SelectV2_1SelectV2dense_1/LessEqual_1:z:0dense_1/add:z:0dense_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_1q
dense_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2_2/e·
dense_1/SelectV2_2SelectV2dense_1/Greater:z:0dense_1/SelectV2_1:output:0dense_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_2p
dense_1/LogLogdense_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Logg
dense_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
dense_1/mul_3/x
dense_1/mul_3Muldense_1/mul_3/x:output:0dense_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_3g
dense_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/add_1/y
dense_1/add_1AddV2dense_1/mul_3:z:0dense_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/add_1£
dense_1/SelectV2_3SelectV2dense_1/Greater_1:z:0dense_1/sub_2:z:0dense_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_3«
dense_1/SelectV2_4SelectV2dense_1/LessEqual:z:0dense_1/mul:z:0dense_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_4í
IdentityIdentitydense_1/SelectV2_4:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
©
b
C__inference_dropout_layer_call_and_return_conditional_losses_402385

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
£
F__inference_sequential_layer_call_and_return_conditional_losses_402344

inputs
dense_402277:
dense_402279: 
dense_1_402338:
dense_1_402340:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_402277dense_402279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4022762
dense/StatefulPartitionedCallñ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4022872
dropout/PartitionedCall©
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_402338dense_1_402340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4023372!
dense_1/StatefulPartitionedCall¾
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÁW
ß
"__inference__traced_restore_403235
file_prefix/
assignvariableop_dense_kernel:+
assignvariableop_1_dense_bias:3
!assignvariableop_2_dense_1_kernel:-
assignvariableop_3_dense_1_bias:'
assignvariableop_4_nadam_iter:	 )
assignvariableop_5_nadam_beta_1: )
assignvariableop_6_nadam_beta_2: (
assignvariableop_7_nadam_decay: 0
&assignvariableop_8_nadam_learning_rate: 1
'assignvariableop_9_nadam_momentum_cache: #
assignvariableop_10_total: #
assignvariableop_11_count: :
(assignvariableop_12_nadam_dense_kernel_m:4
&assignvariableop_13_nadam_dense_bias_m:<
*assignvariableop_14_nadam_dense_1_kernel_m:6
(assignvariableop_15_nadam_dense_1_bias_m::
(assignvariableop_16_nadam_dense_kernel_v:4
&assignvariableop_17_nadam_dense_bias_v:<
*assignvariableop_18_nadam_dense_1_kernel_v:6
(assignvariableop_19_nadam_dense_1_bias_v:
identity_21¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¡
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*­

value£
B 
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¢
AssignVariableOp_4AssignVariableOpassignvariableop_4_nadam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_nadam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8«
AssignVariableOp_8AssignVariableOp&assignvariableop_8_nadam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¬
AssignVariableOp_9AssignVariableOp'assignvariableop_9_nadam_momentum_cacheIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOp(assignvariableop_12_nadam_dense_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13®
AssignVariableOp_13AssignVariableOp&assignvariableop_13_nadam_dense_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14²
AssignVariableOp_14AssignVariableOp*assignvariableop_14_nadam_dense_1_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15°
AssignVariableOp_15AssignVariableOp(assignvariableop_15_nadam_dense_1_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16°
AssignVariableOp_16AssignVariableOp(assignvariableop_16_nadam_dense_kernel_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17®
AssignVariableOp_17AssignVariableOp&assignvariableop_17_nadam_dense_bias_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18²
AssignVariableOp_18AssignVariableOp*assignvariableop_18_nadam_dense_1_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19°
AssignVariableOp_19AssignVariableOp(assignvariableop_19_nadam_dense_1_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
(
ô
C__inference_dense_1_layer_call_and_return_conditional_losses_402337

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constw
	LessEqual	LessEqualBiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	LessEqualW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1s
GreaterGreaterBiasAdd:output:0Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
GreaterW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2	
Const_2w
	Greater_1GreaterBiasAdd:output:0Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Greater_1W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2	
Const_3}
LessEqual_1	LessEqualBiasAdd:output:0Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LessEqual_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/ye
subSubBiasAdd:output:0sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e
SelectV2SelectV2LessEqual:z:0sub:z:0SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2V
ExpExpSelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ExpS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
mul/x\
mulMulmul/x:output:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2	
mul_1/xk
mul_1Mulmul_1/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2	
sub_1/yd
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_1[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xn
truedivRealDivtruediv/x:output:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truedivW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xf
sub_2Subsub_2/x:output:0truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_2W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2	
mul_2/xk
mul_2Mulmul_2/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/y`
addAddV2	mul_2:z:0add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adda
SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SelectV2_1/e

SelectV2_1SelectV2LessEqual_1:z:0add:z:0SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_1a
SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SelectV2_2/e

SelectV2_2SelectV2Greater:z:0SelectV2_1:output:0SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_2X
LogLogSelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LogW
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2	
mul_3/xb
mul_3Mulmul_3/x:output:0Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yf
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1{

SelectV2_3SelectV2Greater_1:z:0	sub_2:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_3

SelectV2_4SelectV2LessEqual:z:0mul:z:0SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_4
IdentityIdentitySelectV2_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬g
Æ
F__inference_sequential_layer_call_and_return_conditional_losses_402941
dense_input6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense_input#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd_
dense/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/Const
dense/LessEqual	LessEqualdense/BiasAdd:output:0dense/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/LessEqualc
dense/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/Const_1
dense/GreaterGreaterdense/BiasAdd:output:0dense/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Greaterc
dense/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense/Const_2
dense/Greater_1Greaterdense/BiasAdd:output:0dense/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Greater_1c
dense/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense/Const_3
dense/LessEqual_1	LessEqualdense/BiasAdd:output:0dense/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/LessEqual_1_
dense/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/sub/y}
	dense/subSubdense/BiasAdd:output:0dense/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/subi
dense/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2/e
dense/SelectV2SelectV2dense/LessEqual:z:0dense/sub:z:0dense/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2h
	dense/ExpExpdense/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/Exp_
dense/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
dense/mul/xt
	dense/mulMuldense/mul/x:output:0dense/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/mulc
dense/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
dense/mul_1/x
dense/mul_1Muldense/mul_1/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_1c
dense/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
dense/sub_1/y|
dense/sub_1Subdense/mul_1:z:0dense/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/sub_1g
dense/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/truediv/x
dense/truedivRealDivdense/truediv/x:output:0dense/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/truedivc
dense/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/sub_2/x~
dense/sub_2Subdense/sub_2/x:output:0dense/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/sub_2c
dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
dense/mul_2/x
dense/mul_2Muldense/mul_2/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_2_
dense/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/add/yx
	dense/addAddV2dense/mul_2:z:0dense/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/addm
dense/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2_1/e¥
dense/SelectV2_1SelectV2dense/LessEqual_1:z:0dense/add:z:0dense/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_1m
dense/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2_2/e­
dense/SelectV2_2SelectV2dense/Greater:z:0dense/SelectV2_1:output:0dense/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_2j
	dense/LogLogdense/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/Logc
dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
dense/mul_3/xz
dense/mul_3Muldense/mul_3/x:output:0dense/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_3c
dense/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/add_1/y~
dense/add_1AddV2dense/mul_3:z:0dense/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/add_1
dense/SelectV2_3SelectV2dense/Greater_1:z:0dense/sub_2:z:0dense/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_3¡
dense/SelectV2_4SelectV2dense/LessEqual:z:0dense/mul:z:0dense/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_4s
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/dropout/Const
dropout/dropout/MulMuldense/SelectV2_4:output:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapedense/SelectV2_4:output:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92 
dropout/dropout/GreaterEqual/yÞ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Mul_1¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddc
dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/Const
dense_1/LessEqual	LessEqualdense_1/BiasAdd:output:0dense_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/LessEqualg
dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/Const_1
dense_1/GreaterGreaterdense_1/BiasAdd:output:0dense_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Greaterg
dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense_1/Const_2
dense_1/Greater_1Greaterdense_1/BiasAdd:output:0dense_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Greater_1g
dense_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense_1/Const_3
dense_1/LessEqual_1	LessEqualdense_1/BiasAdd:output:0dense_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/LessEqual_1c
dense_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/sub/y
dense_1/subSubdense_1/BiasAdd:output:0dense_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/subm
dense_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2/e§
dense_1/SelectV2SelectV2dense_1/LessEqual:z:0dense_1/sub:z:0dense_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2n
dense_1/ExpExpdense_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Expc
dense_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
dense_1/mul/x|
dense_1/mulMuldense_1/mul/x:output:0dense_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mulg
dense_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
dense_1/mul_1/x
dense_1/mul_1Muldense_1/mul_1/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_1g
dense_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
dense_1/sub_1/y
dense_1/sub_1Subdense_1/mul_1:z:0dense_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/sub_1k
dense_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/truediv/x
dense_1/truedivRealDivdense_1/truediv/x:output:0dense_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/truedivg
dense_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/sub_2/x
dense_1/sub_2Subdense_1/sub_2/x:output:0dense_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/sub_2g
dense_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
dense_1/mul_2/x
dense_1/mul_2Muldense_1/mul_2/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_2c
dense_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/add/y
dense_1/addAddV2dense_1/mul_2:z:0dense_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/addq
dense_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2_1/e¯
dense_1/SelectV2_1SelectV2dense_1/LessEqual_1:z:0dense_1/add:z:0dense_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_1q
dense_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2_2/e·
dense_1/SelectV2_2SelectV2dense_1/Greater:z:0dense_1/SelectV2_1:output:0dense_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_2p
dense_1/LogLogdense_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Logg
dense_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
dense_1/mul_3/x
dense_1/mul_3Muldense_1/mul_3/x:output:0dense_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_3g
dense_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/add_1/y
dense_1/add_1AddV2dense_1/mul_3:z:0dense_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/add_1£
dense_1/SelectV2_3SelectV2dense_1/Greater_1:z:0dense_1/sub_2:z:0dense_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_3«
dense_1/SelectV2_4SelectV2dense_1/LessEqual:z:0dense_1/mul:z:0dense_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_4í
IdentityIdentitydense_1/SelectV2_4:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
²w
ù
!__inference__wrapped_model_402221
dense_inputA
/sequential_dense_matmul_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOpÀ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&sequential/dense/MatMul/ReadVariableOp«
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/MatMul¿
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/BiasAddu
sequential/dense/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/dense/Const»
sequential/dense/LessEqual	LessEqual!sequential/dense/BiasAdd:output:0sequential/dense/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/LessEqualy
sequential/dense/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/dense/Const_1·
sequential/dense/GreaterGreater!sequential/dense/BiasAdd:output:0!sequential/dense/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/Greatery
sequential/dense/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
sequential/dense/Const_2»
sequential/dense/Greater_1Greater!sequential/dense/BiasAdd:output:0!sequential/dense/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/Greater_1y
sequential/dense/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
sequential/dense/Const_3Á
sequential/dense/LessEqual_1	LessEqual!sequential/dense/BiasAdd:output:0!sequential/dense/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/LessEqual_1u
sequential/dense/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/dense/sub/y©
sequential/dense/subSub!sequential/dense/BiasAdd:output:0sequential/dense/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/sub
sequential/dense/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/dense/SelectV2/eÔ
sequential/dense/SelectV2SelectV2sequential/dense/LessEqual:z:0sequential/dense/sub:z:0$sequential/dense/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/SelectV2
sequential/dense/ExpExp"sequential/dense/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/Expu
sequential/dense/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
sequential/dense/mul/x 
sequential/dense/mulMulsequential/dense/mul/x:output:0sequential/dense/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/muly
sequential/dense/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
sequential/dense/mul_1/x¯
sequential/dense/mul_1Mul!sequential/dense/mul_1/x:output:0!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/mul_1y
sequential/dense/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
sequential/dense/sub_1/y¨
sequential/dense/sub_1Subsequential/dense/mul_1:z:0!sequential/dense/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/sub_1}
sequential/dense/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/dense/truediv/x²
sequential/dense/truedivRealDiv#sequential/dense/truediv/x:output:0sequential/dense/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/truedivy
sequential/dense/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/dense/sub_2/xª
sequential/dense/sub_2Sub!sequential/dense/sub_2/x:output:0sequential/dense/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/sub_2y
sequential/dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
sequential/dense/mul_2/x¯
sequential/dense/mul_2Mul!sequential/dense/mul_2/x:output:0!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/mul_2u
sequential/dense/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/dense/add/y¤
sequential/dense/addAddV2sequential/dense/mul_2:z:0sequential/dense/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/add
sequential/dense/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/dense/SelectV2_1/eÜ
sequential/dense/SelectV2_1SelectV2 sequential/dense/LessEqual_1:z:0sequential/dense/add:z:0&sequential/dense/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/SelectV2_1
sequential/dense/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/dense/SelectV2_2/eä
sequential/dense/SelectV2_2SelectV2sequential/dense/Greater:z:0$sequential/dense/SelectV2_1:output:0&sequential/dense/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/SelectV2_2
sequential/dense/LogLog$sequential/dense/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/Logy
sequential/dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
sequential/dense/mul_3/x¦
sequential/dense/mul_3Mul!sequential/dense/mul_3/x:output:0sequential/dense/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/mul_3y
sequential/dense/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/dense/add_1/yª
sequential/dense/add_1AddV2sequential/dense/mul_3:z:0!sequential/dense/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/add_1Ð
sequential/dense/SelectV2_3SelectV2sequential/dense/Greater_1:z:0sequential/dense/sub_2:z:0sequential/dense/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/SelectV2_3Ø
sequential/dense/SelectV2_4SelectV2sequential/dense/LessEqual:z:0sequential/dense/mul:z:0$sequential/dense/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/SelectV2_4
sequential/dropout/IdentityIdentity$sequential/dense/SelectV2_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dropout/IdentityÆ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpÊ
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/MatMulÅ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpÍ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/BiasAddy
sequential/dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/dense_1/ConstÃ
sequential/dense_1/LessEqual	LessEqual#sequential/dense_1/BiasAdd:output:0!sequential/dense_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/LessEqual}
sequential/dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/dense_1/Const_1¿
sequential/dense_1/GreaterGreater#sequential/dense_1/BiasAdd:output:0#sequential/dense_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/Greater}
sequential/dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
sequential/dense_1/Const_2Ã
sequential/dense_1/Greater_1Greater#sequential/dense_1/BiasAdd:output:0#sequential/dense_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/Greater_1}
sequential/dense_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
sequential/dense_1/Const_3É
sequential/dense_1/LessEqual_1	LessEqual#sequential/dense_1/BiasAdd:output:0#sequential/dense_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential/dense_1/LessEqual_1y
sequential/dense_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/dense_1/sub/y±
sequential/dense_1/subSub#sequential/dense_1/BiasAdd:output:0!sequential/dense_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/sub
sequential/dense_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/dense_1/SelectV2/eÞ
sequential/dense_1/SelectV2SelectV2 sequential/dense_1/LessEqual:z:0sequential/dense_1/sub:z:0&sequential/dense_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/SelectV2
sequential/dense_1/ExpExp$sequential/dense_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/Expy
sequential/dense_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
sequential/dense_1/mul/x¨
sequential/dense_1/mulMul!sequential/dense_1/mul/x:output:0sequential/dense_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/mul}
sequential/dense_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
sequential/dense_1/mul_1/x·
sequential/dense_1/mul_1Mul#sequential/dense_1/mul_1/x:output:0#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/mul_1}
sequential/dense_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
sequential/dense_1/sub_1/y°
sequential/dense_1/sub_1Subsequential/dense_1/mul_1:z:0#sequential/dense_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/sub_1
sequential/dense_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/dense_1/truediv/xº
sequential/dense_1/truedivRealDiv%sequential/dense_1/truediv/x:output:0sequential/dense_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/truediv}
sequential/dense_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/dense_1/sub_2/x²
sequential/dense_1/sub_2Sub#sequential/dense_1/sub_2/x:output:0sequential/dense_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/sub_2}
sequential/dense_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
sequential/dense_1/mul_2/x·
sequential/dense_1/mul_2Mul#sequential/dense_1/mul_2/x:output:0#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/mul_2y
sequential/dense_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/dense_1/add/y¬
sequential/dense_1/addAddV2sequential/dense_1/mul_2:z:0!sequential/dense_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/add
sequential/dense_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/dense_1/SelectV2_1/eæ
sequential/dense_1/SelectV2_1SelectV2"sequential/dense_1/LessEqual_1:z:0sequential/dense_1/add:z:0(sequential/dense_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/SelectV2_1
sequential/dense_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/dense_1/SelectV2_2/eî
sequential/dense_1/SelectV2_2SelectV2sequential/dense_1/Greater:z:0&sequential/dense_1/SelectV2_1:output:0(sequential/dense_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/SelectV2_2
sequential/dense_1/LogLog&sequential/dense_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/Log}
sequential/dense_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
sequential/dense_1/mul_3/x®
sequential/dense_1/mul_3Mul#sequential/dense_1/mul_3/x:output:0sequential/dense_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/mul_3}
sequential/dense_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/dense_1/add_1/y²
sequential/dense_1/add_1AddV2sequential/dense_1/mul_3:z:0#sequential/dense_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/add_1Ú
sequential/dense_1/SelectV2_3SelectV2 sequential/dense_1/Greater_1:z:0sequential/dense_1/sub_2:z:0sequential/dense_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/SelectV2_3â
sequential/dense_1/SelectV2_4SelectV2 sequential/dense_1/LessEqual:z:0sequential/dense_1/mul:z:0&sequential/dense_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/SelectV2_4¤
IdentityIdentity&sequential/dense_1/SelectV2_4:output:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input

Ó
+__inference_sequential_layer_call_fn_402516
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4023442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
ð
a
C__inference_dropout_layer_call_and_return_conditional_losses_403013

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
(
ò
A__inference_dense_layer_call_and_return_conditional_losses_402276

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constw
	LessEqual	LessEqualBiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	LessEqualW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1s
GreaterGreaterBiasAdd:output:0Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
GreaterW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2	
Const_2w
	Greater_1GreaterBiasAdd:output:0Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Greater_1W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2	
Const_3}
LessEqual_1	LessEqualBiasAdd:output:0Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LessEqual_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/ye
subSubBiasAdd:output:0sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e
SelectV2SelectV2LessEqual:z:0sub:z:0SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2V
ExpExpSelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ExpS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
mul/x\
mulMulmul/x:output:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2	
mul_1/xk
mul_1Mulmul_1/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2	
sub_1/yd
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_1[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xn
truedivRealDivtruediv/x:output:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truedivW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xf
sub_2Subsub_2/x:output:0truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_2W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2	
mul_2/xk
mul_2Mulmul_2/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/y`
addAddV2	mul_2:z:0add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adda
SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SelectV2_1/e

SelectV2_1SelectV2LessEqual_1:z:0add:z:0SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_1a
SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SelectV2_2/e

SelectV2_2SelectV2Greater:z:0SelectV2_1:output:0SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_2X
LogLogSelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LogW
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2	
mul_3/xb
mul_3Mulmul_3/x:output:0Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yf
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1{

SelectV2_3SelectV2Greater_1:z:0	sub_2:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_3

SelectV2_4SelectV2LessEqual:z:0mul:z:0SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_4
IdentityIdentitySelectV2_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
Î
+__inference_sequential_layer_call_fn_402529

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4023442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
a
(__inference_dropout_layer_call_fn_403008

inputs
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4023852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
(
ô
C__inference_dense_1_layer_call_and_return_conditional_losses_403082

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constw
	LessEqual	LessEqualBiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	LessEqualW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1s
GreaterGreaterBiasAdd:output:0Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
GreaterW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2	
Const_2w
	Greater_1GreaterBiasAdd:output:0Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Greater_1W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2	
Const_3}
LessEqual_1	LessEqualBiasAdd:output:0Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LessEqual_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/ye
subSubBiasAdd:output:0sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e
SelectV2SelectV2LessEqual:z:0sub:z:0SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2V
ExpExpSelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ExpS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
mul/x\
mulMulmul/x:output:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2	
mul_1/xk
mul_1Mulmul_1/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2	
sub_1/yd
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_1[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xn
truedivRealDivtruediv/x:output:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truedivW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xf
sub_2Subsub_2/x:output:0truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_2W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2	
mul_2/xk
mul_2Mulmul_2/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/y`
addAddV2	mul_2:z:0add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adda
SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SelectV2_1/e

SelectV2_1SelectV2LessEqual_1:z:0add:z:0SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_1a
SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SelectV2_2/e

SelectV2_2SelectV2Greater:z:0SelectV2_1:output:0SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_2X
LogLogSelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LogW
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2	
mul_3/xb
mul_3Mulmul_3/x:output:0Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yf
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1{

SelectV2_3SelectV2Greater_1:z:0	sub_2:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_3

SelectV2_4SelectV2LessEqual:z:0mul:z:0SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_4
IdentityIdentitySelectV2_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
g
Á
F__inference_sequential_layer_call_and_return_conditional_losses_402748

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd_
dense/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/Const
dense/LessEqual	LessEqualdense/BiasAdd:output:0dense/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/LessEqualc
dense/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/Const_1
dense/GreaterGreaterdense/BiasAdd:output:0dense/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Greaterc
dense/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense/Const_2
dense/Greater_1Greaterdense/BiasAdd:output:0dense/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Greater_1c
dense/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense/Const_3
dense/LessEqual_1	LessEqualdense/BiasAdd:output:0dense/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/LessEqual_1_
dense/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/sub/y}
	dense/subSubdense/BiasAdd:output:0dense/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/subi
dense/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2/e
dense/SelectV2SelectV2dense/LessEqual:z:0dense/sub:z:0dense/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2h
	dense/ExpExpdense/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/Exp_
dense/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
dense/mul/xt
	dense/mulMuldense/mul/x:output:0dense/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/mulc
dense/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
dense/mul_1/x
dense/mul_1Muldense/mul_1/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_1c
dense/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
dense/sub_1/y|
dense/sub_1Subdense/mul_1:z:0dense/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/sub_1g
dense/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/truediv/x
dense/truedivRealDivdense/truediv/x:output:0dense/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/truedivc
dense/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/sub_2/x~
dense/sub_2Subdense/sub_2/x:output:0dense/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/sub_2c
dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
dense/mul_2/x
dense/mul_2Muldense/mul_2/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_2_
dense/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense/add/yx
	dense/addAddV2dense/mul_2:z:0dense/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/addm
dense/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2_1/e¥
dense/SelectV2_1SelectV2dense/LessEqual_1:z:0dense/add:z:0dense/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_1m
dense/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/SelectV2_2/e­
dense/SelectV2_2SelectV2dense/Greater:z:0dense/SelectV2_1:output:0dense/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_2j
	dense/LogLogdense/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dense/Logc
dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
dense/mul_3/xz
dense/mul_3Muldense/mul_3/x:output:0dense/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/mul_3c
dense/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense/add_1/y~
dense/add_1AddV2dense/mul_3:z:0dense/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/add_1
dense/SelectV2_3SelectV2dense/Greater_1:z:0dense/sub_2:z:0dense/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_3¡
dense/SelectV2_4SelectV2dense/LessEqual:z:0dense/mul:z:0dense/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/SelectV2_4s
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/dropout/Const
dropout/dropout/MulMuldense/SelectV2_4:output:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapedense/SelectV2_4:output:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92 
dropout/dropout/GreaterEqual/yÞ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Mul_1¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddc
dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/Const
dense_1/LessEqual	LessEqualdense_1/BiasAdd:output:0dense_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/LessEqualg
dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/Const_1
dense_1/GreaterGreaterdense_1/BiasAdd:output:0dense_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Greaterg
dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense_1/Const_2
dense_1/Greater_1Greaterdense_1/BiasAdd:output:0dense_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Greater_1g
dense_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
dense_1/Const_3
dense_1/LessEqual_1	LessEqualdense_1/BiasAdd:output:0dense_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/LessEqual_1c
dense_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/sub/y
dense_1/subSubdense_1/BiasAdd:output:0dense_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/subm
dense_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2/e§
dense_1/SelectV2SelectV2dense_1/LessEqual:z:0dense_1/sub:z:0dense_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2n
dense_1/ExpExpdense_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Expc
dense_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
dense_1/mul/x|
dense_1/mulMuldense_1/mul/x:output:0dense_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mulg
dense_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
dense_1/mul_1/x
dense_1/mul_1Muldense_1/mul_1/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_1g
dense_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
dense_1/sub_1/y
dense_1/sub_1Subdense_1/mul_1:z:0dense_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/sub_1k
dense_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/truediv/x
dense_1/truedivRealDivdense_1/truediv/x:output:0dense_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/truedivg
dense_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/sub_2/x
dense_1/sub_2Subdense_1/sub_2/x:output:0dense_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/sub_2g
dense_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
dense_1/mul_2/x
dense_1/mul_2Muldense_1/mul_2/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_2c
dense_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_1/add/y
dense_1/addAddV2dense_1/mul_2:z:0dense_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/addq
dense_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2_1/e¯
dense_1/SelectV2_1SelectV2dense_1/LessEqual_1:z:0dense_1/add:z:0dense_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_1q
dense_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_1/SelectV2_2/e·
dense_1/SelectV2_2SelectV2dense_1/Greater:z:0dense_1/SelectV2_1:output:0dense_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_2p
dense_1/LogLogdense_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Logg
dense_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
dense_1/mul_3/x
dense_1/mul_3Muldense_1/mul_3/x:output:0dense_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/mul_3g
dense_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/add_1/y
dense_1/add_1AddV2dense_1/mul_3:z:0dense_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/add_1£
dense_1/SelectV2_3SelectV2dense_1/Greater_1:z:0dense_1/sub_2:z:0dense_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_3«
dense_1/SelectV2_4SelectV2dense_1/LessEqual:z:0dense_1/mul:z:0dense_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/SelectV2_4í
IdentityIdentitydense_1/SelectV2_4:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
D
(__inference_dropout_layer_call_fn_403003

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4022872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
Ì
$__inference_signature_wrapper_402503
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_4022212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
ð
a
C__inference_dropout_layer_call_and_return_conditional_losses_402287

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_dense_1_layer_call_fn_403034

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4023372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
(
ò
A__inference_dense_layer_call_and_return_conditional_losses_402998

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constw
	LessEqual	LessEqualBiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	LessEqualW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1s
GreaterGreaterBiasAdd:output:0Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
GreaterW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2	
Const_2w
	Greater_1GreaterBiasAdd:output:0Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Greater_1W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2	
Const_3}
LessEqual_1	LessEqualBiasAdd:output:0Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LessEqual_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/ye
subSubBiasAdd:output:0sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e
SelectV2SelectV2LessEqual:z:0sub:z:0SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2V
ExpExpSelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ExpS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
mul/x\
mulMulmul/x:output:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2	
mul_1/xk
mul_1Mulmul_1/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1W
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2	
sub_1/yd
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_1[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xn
truedivRealDivtruediv/x:output:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truedivW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xf
sub_2Subsub_2/x:output:0truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_2W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2	
mul_2/xk
mul_2Mulmul_2/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/y`
addAddV2	mul_2:z:0add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adda
SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SelectV2_1/e

SelectV2_1SelectV2LessEqual_1:z:0add:z:0SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_1a
SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SelectV2_2/e

SelectV2_2SelectV2Greater:z:0SelectV2_1:output:0SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_2X
LogLogSelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LogW
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2	
mul_3/xb
mul_3Mulmul_3/x:output:0Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yf
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1{

SelectV2_3SelectV2Greater_1:z:0	sub_2:z:0	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_3

SelectV2_4SelectV2LessEqual:z:0mul:z:0SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

SelectV2_4
IdentityIdentitySelectV2_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Å
F__inference_sequential_layer_call_and_return_conditional_losses_402428

inputs
dense_402416:
dense_402418: 
dense_1_402422:
dense_1_402424:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_402416dense_402418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4022762
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4023852!
dropout/StatefulPartitionedCall±
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_402422dense_1_402424*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4023372!
dense_1/StatefulPartitionedCallà
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
Î
+__inference_sequential_layer_call_fn_402542

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4024282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô1
®
__inference__traced_save_403165
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_nadam_dense_kernel_m_read_readvariableop1
-savev2_nadam_dense_bias_m_read_readvariableop5
1savev2_nadam_dense_1_kernel_m_read_readvariableop3
/savev2_nadam_dense_1_bias_m_read_readvariableop3
/savev2_nadam_dense_kernel_v_read_readvariableop1
-savev2_nadam_dense_bias_v_read_readvariableop5
1savev2_nadam_dense_1_kernel_v_read_readvariableop3
/savev2_nadam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*­

value£
B 
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names²
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÀ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_nadam_dense_kernel_m_read_readvariableop-savev2_nadam_dense_bias_m_read_readvariableop1savev2_nadam_dense_1_kernel_m_read_readvariableop/savev2_nadam_dense_1_bias_m_read_readvariableop/savev2_nadam_dense_kernel_v_read_readvariableop-savev2_nadam_dense_bias_v_read_readvariableop1savev2_nadam_dense_1_kernel_v_read_readvariableop/savev2_nadam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapesv
t: ::::: : : : : : : : ::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: "ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
C
dense_input4
serving_default_dense_input:0ÿÿÿÿÿÿÿÿÿ;
dense_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ý|
ê
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
A__call__
*B&call_and_return_all_conditional_losses
C_default_save_signature"Ä
_tf_keras_sequential¥{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "custom_activation", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0002, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "custom_activation", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 9}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [4, 2]}, "float32", "dense_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "custom_activation", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0002, "noise_shape": null, "seed": null}, "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "custom_activation", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7}]}}, "training_config": {"loss": "mean_absolute_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
Ï


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
D__call__
*E&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "custom_activation", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 9}, "build_input_shape": {"class_name": "TensorShape", "items": [4, 2]}}
û
trainable_variables
regularization_losses
	variables
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0002, "noise_shape": null, "seed": null}, "shared_object_id": 4}
Ô

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"¯
_tf_keras_layer{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "custom_activation", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [4, 2]}}
¯
iter

beta_1

beta_2
	decay
learning_rate
momentum_cache
m9m:m;m<
v=v>v?v@"
	optimizer
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
Ê
 layer_regularization_losses

!layers
regularization_losses
"layer_metrics
	variables
#non_trainable_variables
$metrics
trainable_variables
A__call__
C_default_save_signature
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
,
Jserving_default"
signature_map
:2dense/kernel
:2
dense/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­
trainable_variables
%layer_regularization_losses

&layers
regularization_losses
'layer_metrics
	variables
(metrics
)non_trainable_variables
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
*layer_regularization_losses

+layers
regularization_losses
,layer_metrics
	variables
-metrics
.non_trainable_variables
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
trainable_variables
/layer_regularization_losses

0layers
regularization_losses
1layer_metrics
	variables
2metrics
3non_trainable_variables
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
40"
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
Ô
	5total
	6count
7	variables
8	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 11}
:  (2total
:  (2count
.
50
61"
trackable_list_wrapper
-
7	variables"
_generic_user_object
$:"2Nadam/dense/kernel/m
:2Nadam/dense/bias/m
&:$2Nadam/dense_1/kernel/m
 :2Nadam/dense_1/bias/m
$:"2Nadam/dense/kernel/v
:2Nadam/dense/bias/v
&:$2Nadam/dense_1/kernel/v
 :2Nadam/dense_1/bias/v
ú2÷
+__inference_sequential_layer_call_fn_402516
+__inference_sequential_layer_call_fn_402529
+__inference_sequential_layer_call_fn_402542
+__inference_sequential_layer_call_fn_402555À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_402648
F__inference_sequential_layer_call_and_return_conditional_losses_402748
F__inference_sequential_layer_call_and_return_conditional_losses_402841
F__inference_sequential_layer_call_and_return_conditional_losses_402941À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
!__inference__wrapped_model_402221º
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"
dense_inputÿÿÿÿÿÿÿÿÿ
Ð2Í
&__inference_dense_layer_call_fn_402950¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_402998¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
(__inference_dropout_layer_call_fn_403003
(__inference_dropout_layer_call_fn_403008´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á
C__inference_dropout_layer_call_and_return_conditional_losses_403013
C__inference_dropout_layer_call_and_return_conditional_losses_403025´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_dense_1_layer_call_fn_403034¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_403082¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÏBÌ
$__inference_signature_wrapper_402503dense_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!__inference__wrapped_model_402221o
4¢1
*¢'
%"
dense_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_1_layer_call_and_return_conditional_losses_403082\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_1_layer_call_fn_403034O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_layer_call_and_return_conditional_losses_402998\
/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_layer_call_fn_402950O
/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_layer_call_and_return_conditional_losses_403013\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_layer_call_and_return_conditional_losses_403025\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_layer_call_fn_403003O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_layer_call_fn_403008O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ°
F__inference_sequential_layer_call_and_return_conditional_losses_402648f
7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_sequential_layer_call_and_return_conditional_losses_402748f
7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_sequential_layer_call_and_return_conditional_losses_402841k
<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_sequential_layer_call_and_return_conditional_losses_402941k
<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_layer_call_fn_402516^
<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_402529Y
7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_402542Y
7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_402555^
<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¦
$__inference_signature_wrapper_402503~
C¢@
¢ 
9ª6
4
dense_input%"
dense_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ
§	
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718á
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
Ü
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
Ù
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api

iter

 beta_1

!beta_2
	"decay
#learning_rate
$momentum_cachemCmDmEmFvGvHvIvJ
 

0
1
2
3

0
1
2
3
­
regularization_losses

%layers
&layer_regularization_losses
trainable_variables
'layer_metrics
	variables
(metrics
)non_trainable_variables
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses

*layers
+layer_regularization_losses
trainable_variables
,layer_metrics
	variables
-metrics
.non_trainable_variables
 
 
 
­
regularization_losses

/layers
0layer_regularization_losses
trainable_variables
1layer_metrics
	variables
2metrics
3non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses

4layers
5layer_regularization_losses
trainable_variables
6layer_metrics
	variables
7metrics
8non_trainable_variables
 
 
 
­
regularization_losses

9layers
:layer_regularization_losses
trainable_variables
;layer_metrics
	variables
<metrics
=non_trainable_variables
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

0
1
2
3
 
 

>0
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
4
	?total
	@count
A	variables
B	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

A	variables
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
ô
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
GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_2627
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU 2J 8 *&
f!R
__inference__traced_save_3262
þ
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_3332Ö²
£f
¿
D__inference_sequential_layer_call_and_return_conditional_losses_2863

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
dense/BiasAddi
activation/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/Const
activation/LessEqual	LessEqualdense/BiasAdd:output:0activation/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/LessEqualm
activation/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/Const_1
activation/GreaterGreaterdense/BiasAdd:output:0activation/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Greaterm
activation/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation/Const_2
activation/Greater_1Greaterdense/BiasAdd:output:0activation/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Greater_1m
activation/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation/Const_3¤
activation/LessEqual_1	LessEqualdense/BiasAdd:output:0activation/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/LessEqual_1i
activation/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/sub/y
activation/subSubdense/BiasAdd:output:0activation/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/subs
activation/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2/e¶
activation/SelectV2SelectV2activation/LessEqual:z:0activation/sub:z:0activation/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2w
activation/ExpExpactivation/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Expi
activation/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
activation/mul/x
activation/mulMulactivation/mul/x:output:0activation/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mulm
activation/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
activation/mul_1/x
activation/mul_1Mulactivation/mul_1/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_1m
activation/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
activation/sub_1/y
activation/sub_1Subactivation/mul_1:z:0activation/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/sub_1q
activation/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/truediv/x
activation/truedivRealDivactivation/truediv/x:output:0activation/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/truedivm
activation/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/sub_2/x
activation/sub_2Subactivation/sub_2/x:output:0activation/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/sub_2m
activation/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
activation/mul_2/x
activation/mul_2Mulactivation/mul_2/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_2i
activation/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/add/y
activation/addAddV2activation/mul_2:z:0activation/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/addw
activation/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2_1/e¾
activation/SelectV2_1SelectV2activation/LessEqual_1:z:0activation/add:z:0 activation/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_1w
activation/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2_2/eÆ
activation/SelectV2_2SelectV2activation/Greater:z:0activation/SelectV2_1:output:0 activation/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_2y
activation/LogLogactivation/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Logm
activation/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
activation/mul_3/x
activation/mul_3Mulactivation/mul_3/x:output:0activation/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_3m
activation/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/add_1/y
activation/add_1AddV2activation/mul_3:z:0activation/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/add_1²
activation/SelectV2_3SelectV2activation/Greater_1:z:0activation/sub_2:z:0activation/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_3º
activation/SelectV2_4SelectV2activation/LessEqual:z:0activation/mul:z:0activation/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_4¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp£
dense_1/MatMulMatMulactivation/SelectV2_4:output:0%dense_1/MatMul/ReadVariableOp:value:0*
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
dense_1/BiasAddm
activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/Const¦
activation_1/LessEqual	LessEqualdense_1/BiasAdd:output:0activation_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/LessEqualq
activation_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/Const_1¢
activation_1/GreaterGreaterdense_1/BiasAdd:output:0activation_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Greaterq
activation_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation_1/Const_2¦
activation_1/Greater_1Greaterdense_1/BiasAdd:output:0activation_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Greater_1q
activation_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation_1/Const_3¬
activation_1/LessEqual_1	LessEqualdense_1/BiasAdd:output:0activation_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/LessEqual_1m
activation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/sub/y
activation_1/subSubdense_1/BiasAdd:output:0activation_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/subw
activation_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2/eÀ
activation_1/SelectV2SelectV2activation_1/LessEqual:z:0activation_1/sub:z:0 activation_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2}
activation_1/ExpExpactivation_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Expm
activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
activation_1/mul/x
activation_1/mulMulactivation_1/mul/x:output:0activation_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mulq
activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
activation_1/mul_1/x
activation_1/mul_1Mulactivation_1/mul_1/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_1q
activation_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
activation_1/sub_1/y
activation_1/sub_1Subactivation_1/mul_1:z:0activation_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/sub_1u
activation_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/truediv/x¢
activation_1/truedivRealDivactivation_1/truediv/x:output:0activation_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/truedivq
activation_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/sub_2/x
activation_1/sub_2Subactivation_1/sub_2/x:output:0activation_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/sub_2q
activation_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
activation_1/mul_2/x
activation_1/mul_2Mulactivation_1/mul_2/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_2m
activation_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/add/y
activation_1/addAddV2activation_1/mul_2:z:0activation_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/add{
activation_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2_1/eÈ
activation_1/SelectV2_1SelectV2activation_1/LessEqual_1:z:0activation_1/add:z:0"activation_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_1{
activation_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2_2/eÐ
activation_1/SelectV2_2SelectV2activation_1/Greater:z:0 activation_1/SelectV2_1:output:0"activation_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_2
activation_1/LogLog activation_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Logq
activation_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
activation_1/mul_3/x
activation_1/mul_3Mulactivation_1/mul_3/x:output:0activation_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_3q
activation_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_1/add_1/y
activation_1/add_1AddV2activation_1/mul_3:z:0activation_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/add_1¼
activation_1/SelectV2_3SelectV2activation_1/Greater_1:z:0activation_1/sub_2:z:0activation_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_3Ä
activation_1/SelectV2_4SelectV2activation_1/LessEqual:z:0activation_1/mul:z:0 activation_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_4ò
IdentityIdentity activation_1/SelectV2_4:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
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
¿W
Ý
 __inference__traced_restore_3332
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
Ñ

D__inference_sequential_layer_call_and_return_conditional_losses_2476

inputs

dense_2366:

dense_2368:
dense_1_2426:
dense_1_2428:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallÿ
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_2366
dense_2368*
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
GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_23652
dense/StatefulPartitionedCallø
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_24132
activation/PartitionedCall¦
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_2426dense_1_2428*
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
GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24252!
dense_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_24732
activation_1/PartitionedCall»
IdentityIdentity%activation_1/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
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
þ
Ñ
)__inference_sequential_layer_call_fn_2679
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
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
GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_25502
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
£f
¿
D__inference_sequential_layer_call_and_return_conditional_losses_2771

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
dense/BiasAddi
activation/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/Const
activation/LessEqual	LessEqualdense/BiasAdd:output:0activation/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/LessEqualm
activation/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/Const_1
activation/GreaterGreaterdense/BiasAdd:output:0activation/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Greaterm
activation/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation/Const_2
activation/Greater_1Greaterdense/BiasAdd:output:0activation/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Greater_1m
activation/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation/Const_3¤
activation/LessEqual_1	LessEqualdense/BiasAdd:output:0activation/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/LessEqual_1i
activation/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/sub/y
activation/subSubdense/BiasAdd:output:0activation/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/subs
activation/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2/e¶
activation/SelectV2SelectV2activation/LessEqual:z:0activation/sub:z:0activation/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2w
activation/ExpExpactivation/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Expi
activation/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
activation/mul/x
activation/mulMulactivation/mul/x:output:0activation/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mulm
activation/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
activation/mul_1/x
activation/mul_1Mulactivation/mul_1/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_1m
activation/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
activation/sub_1/y
activation/sub_1Subactivation/mul_1:z:0activation/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/sub_1q
activation/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/truediv/x
activation/truedivRealDivactivation/truediv/x:output:0activation/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/truedivm
activation/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/sub_2/x
activation/sub_2Subactivation/sub_2/x:output:0activation/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/sub_2m
activation/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
activation/mul_2/x
activation/mul_2Mulactivation/mul_2/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_2i
activation/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/add/y
activation/addAddV2activation/mul_2:z:0activation/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/addw
activation/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2_1/e¾
activation/SelectV2_1SelectV2activation/LessEqual_1:z:0activation/add:z:0 activation/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_1w
activation/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2_2/eÆ
activation/SelectV2_2SelectV2activation/Greater:z:0activation/SelectV2_1:output:0 activation/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_2y
activation/LogLogactivation/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Logm
activation/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
activation/mul_3/x
activation/mul_3Mulactivation/mul_3/x:output:0activation/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_3m
activation/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/add_1/y
activation/add_1AddV2activation/mul_3:z:0activation/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/add_1²
activation/SelectV2_3SelectV2activation/Greater_1:z:0activation/sub_2:z:0activation/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_3º
activation/SelectV2_4SelectV2activation/LessEqual:z:0activation/mul:z:0activation/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_4¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp£
dense_1/MatMulMatMulactivation/SelectV2_4:output:0%dense_1/MatMul/ReadVariableOp:value:0*
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
dense_1/BiasAddm
activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/Const¦
activation_1/LessEqual	LessEqualdense_1/BiasAdd:output:0activation_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/LessEqualq
activation_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/Const_1¢
activation_1/GreaterGreaterdense_1/BiasAdd:output:0activation_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Greaterq
activation_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation_1/Const_2¦
activation_1/Greater_1Greaterdense_1/BiasAdd:output:0activation_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Greater_1q
activation_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation_1/Const_3¬
activation_1/LessEqual_1	LessEqualdense_1/BiasAdd:output:0activation_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/LessEqual_1m
activation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/sub/y
activation_1/subSubdense_1/BiasAdd:output:0activation_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/subw
activation_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2/eÀ
activation_1/SelectV2SelectV2activation_1/LessEqual:z:0activation_1/sub:z:0 activation_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2}
activation_1/ExpExpactivation_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Expm
activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
activation_1/mul/x
activation_1/mulMulactivation_1/mul/x:output:0activation_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mulq
activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
activation_1/mul_1/x
activation_1/mul_1Mulactivation_1/mul_1/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_1q
activation_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
activation_1/sub_1/y
activation_1/sub_1Subactivation_1/mul_1:z:0activation_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/sub_1u
activation_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/truediv/x¢
activation_1/truedivRealDivactivation_1/truediv/x:output:0activation_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/truedivq
activation_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/sub_2/x
activation_1/sub_2Subactivation_1/sub_2/x:output:0activation_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/sub_2q
activation_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
activation_1/mul_2/x
activation_1/mul_2Mulactivation_1/mul_2/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_2m
activation_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/add/y
activation_1/addAddV2activation_1/mul_2:z:0activation_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/add{
activation_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2_1/eÈ
activation_1/SelectV2_1SelectV2activation_1/LessEqual_1:z:0activation_1/add:z:0"activation_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_1{
activation_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2_2/eÐ
activation_1/SelectV2_2SelectV2activation_1/Greater:z:0 activation_1/SelectV2_1:output:0"activation_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_2
activation_1/LogLog activation_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Logq
activation_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
activation_1/mul_3/x
activation_1/mul_3Mulactivation_1/mul_3/x:output:0activation_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_3q
activation_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_1/add_1/y
activation_1/add_1AddV2activation_1/mul_3:z:0activation_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/add_1¼
activation_1/SelectV2_3SelectV2activation_1/Greater_1:z:0activation_1/sub_2:z:0activation_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_3Ä
activation_1/SelectV2_4SelectV2activation_1/LessEqual:z:0activation_1/mul:z:0 activation_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_4ò
IdentityIdentity activation_1/SelectV2_4:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
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
þ
Ñ
)__inference_sequential_layer_call_fn_2640
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
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
GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_24762
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
!
`
D__inference_activation_layer_call_and_return_conditional_losses_3113

inputs
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
	LessEqual	LessEqualinputsConst:output:0*
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
Const_1i
GreaterGreaterinputsConst_1:output:0*
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
Const_2m
	Greater_1GreaterinputsConst_2:output:0*
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
Const_3s
LessEqual_1	LessEqualinputsConst_3:output:0*
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
sub/y[
subSubinputssub/y:output:0*
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
mul_1/xa
mul_1Mulmul_1/x:output:0inputs*
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
mul_2/xa
mul_2Mulmul_2/x:output:0inputs*
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

SelectV2_4g
IdentityIdentitySelectV2_4:output:0*
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
Ò
Ê
"__inference_signature_wrapper_2627
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallî
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
GPU 2J 8 *(
f#R!
__inference__wrapped_model_23482
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
ï
Ì
)__inference_sequential_layer_call_fn_2666

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
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
GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_25502
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


&__inference_dense_1_layer_call_fn_3122

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallñ
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
GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24252
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
Í	
ò
A__inference_dense_1_layer_call_and_return_conditional_losses_3132

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
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
Ò1
¬
__inference__traced_save_3262
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
: 
ï
Ì
)__inference_sequential_layer_call_fn_2653

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
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
GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_24762
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
²f
Ä
D__inference_sequential_layer_call_and_return_conditional_losses_3047
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
dense/BiasAddi
activation/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/Const
activation/LessEqual	LessEqualdense/BiasAdd:output:0activation/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/LessEqualm
activation/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/Const_1
activation/GreaterGreaterdense/BiasAdd:output:0activation/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Greaterm
activation/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation/Const_2
activation/Greater_1Greaterdense/BiasAdd:output:0activation/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Greater_1m
activation/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation/Const_3¤
activation/LessEqual_1	LessEqualdense/BiasAdd:output:0activation/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/LessEqual_1i
activation/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/sub/y
activation/subSubdense/BiasAdd:output:0activation/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/subs
activation/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2/e¶
activation/SelectV2SelectV2activation/LessEqual:z:0activation/sub:z:0activation/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2w
activation/ExpExpactivation/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Expi
activation/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
activation/mul/x
activation/mulMulactivation/mul/x:output:0activation/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mulm
activation/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
activation/mul_1/x
activation/mul_1Mulactivation/mul_1/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_1m
activation/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
activation/sub_1/y
activation/sub_1Subactivation/mul_1:z:0activation/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/sub_1q
activation/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/truediv/x
activation/truedivRealDivactivation/truediv/x:output:0activation/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/truedivm
activation/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/sub_2/x
activation/sub_2Subactivation/sub_2/x:output:0activation/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/sub_2m
activation/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
activation/mul_2/x
activation/mul_2Mulactivation/mul_2/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_2i
activation/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/add/y
activation/addAddV2activation/mul_2:z:0activation/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/addw
activation/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2_1/e¾
activation/SelectV2_1SelectV2activation/LessEqual_1:z:0activation/add:z:0 activation/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_1w
activation/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2_2/eÆ
activation/SelectV2_2SelectV2activation/Greater:z:0activation/SelectV2_1:output:0 activation/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_2y
activation/LogLogactivation/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Logm
activation/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
activation/mul_3/x
activation/mul_3Mulactivation/mul_3/x:output:0activation/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_3m
activation/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/add_1/y
activation/add_1AddV2activation/mul_3:z:0activation/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/add_1²
activation/SelectV2_3SelectV2activation/Greater_1:z:0activation/sub_2:z:0activation/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_3º
activation/SelectV2_4SelectV2activation/LessEqual:z:0activation/mul:z:0activation/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_4¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp£
dense_1/MatMulMatMulactivation/SelectV2_4:output:0%dense_1/MatMul/ReadVariableOp:value:0*
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
dense_1/BiasAddm
activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/Const¦
activation_1/LessEqual	LessEqualdense_1/BiasAdd:output:0activation_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/LessEqualq
activation_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/Const_1¢
activation_1/GreaterGreaterdense_1/BiasAdd:output:0activation_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Greaterq
activation_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation_1/Const_2¦
activation_1/Greater_1Greaterdense_1/BiasAdd:output:0activation_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Greater_1q
activation_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation_1/Const_3¬
activation_1/LessEqual_1	LessEqualdense_1/BiasAdd:output:0activation_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/LessEqual_1m
activation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/sub/y
activation_1/subSubdense_1/BiasAdd:output:0activation_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/subw
activation_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2/eÀ
activation_1/SelectV2SelectV2activation_1/LessEqual:z:0activation_1/sub:z:0 activation_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2}
activation_1/ExpExpactivation_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Expm
activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
activation_1/mul/x
activation_1/mulMulactivation_1/mul/x:output:0activation_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mulq
activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
activation_1/mul_1/x
activation_1/mul_1Mulactivation_1/mul_1/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_1q
activation_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
activation_1/sub_1/y
activation_1/sub_1Subactivation_1/mul_1:z:0activation_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/sub_1u
activation_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/truediv/x¢
activation_1/truedivRealDivactivation_1/truediv/x:output:0activation_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/truedivq
activation_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/sub_2/x
activation_1/sub_2Subactivation_1/sub_2/x:output:0activation_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/sub_2q
activation_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
activation_1/mul_2/x
activation_1/mul_2Mulactivation_1/mul_2/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_2m
activation_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/add/y
activation_1/addAddV2activation_1/mul_2:z:0activation_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/add{
activation_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2_1/eÈ
activation_1/SelectV2_1SelectV2activation_1/LessEqual_1:z:0activation_1/add:z:0"activation_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_1{
activation_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2_2/eÐ
activation_1/SelectV2_2SelectV2activation_1/Greater:z:0 activation_1/SelectV2_1:output:0"activation_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_2
activation_1/LogLog activation_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Logq
activation_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
activation_1/mul_3/x
activation_1/mul_3Mulactivation_1/mul_3/x:output:0activation_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_3q
activation_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_1/add_1/y
activation_1/add_1AddV2activation_1/mul_3:z:0activation_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/add_1¼
activation_1/SelectV2_3SelectV2activation_1/Greater_1:z:0activation_1/sub_2:z:0activation_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_3Ä
activation_1/SelectV2_4SelectV2activation_1/LessEqual:z:0activation_1/mul:z:0 activation_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_4ò
IdentityIdentity activation_1/SelectV2_4:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
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
¼
E
)__inference_activation_layer_call_fn_3071

inputs
identityÂ
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
GPU 2J 8 *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_24132
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


$__inference_dense_layer_call_fn_3056

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallï
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
GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_23652
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
¥
÷
__inference__wrapped_model_2348
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
sequential/dense/BiasAdd
sequential/activation/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/activation/ConstÊ
sequential/activation/LessEqual	LessEqual!sequential/dense/BiasAdd:output:0$sequential/activation/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/activation/LessEqual
sequential/activation/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/activation/Const_1Æ
sequential/activation/GreaterGreater!sequential/dense/BiasAdd:output:0&sequential/activation/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/Greater
sequential/activation/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
sequential/activation/Const_2Ê
sequential/activation/Greater_1Greater!sequential/dense/BiasAdd:output:0&sequential/activation/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/activation/Greater_1
sequential/activation/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
sequential/activation/Const_3Ð
!sequential/activation/LessEqual_1	LessEqual!sequential/dense/BiasAdd:output:0&sequential/activation/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/activation/LessEqual_1
sequential/activation/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/activation/sub/y¸
sequential/activation/subSub!sequential/dense/BiasAdd:output:0$sequential/activation/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/sub
 sequential/activation/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential/activation/SelectV2/eí
sequential/activation/SelectV2SelectV2#sequential/activation/LessEqual:z:0sequential/activation/sub:z:0)sequential/activation/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential/activation/SelectV2
sequential/activation/ExpExp'sequential/activation/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/Exp
sequential/activation/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
sequential/activation/mul/x´
sequential/activation/mulMul$sequential/activation/mul/x:output:0sequential/activation/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/mul
sequential/activation/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
sequential/activation/mul_1/x¾
sequential/activation/mul_1Mul&sequential/activation/mul_1/x:output:0!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/mul_1
sequential/activation/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
sequential/activation/sub_1/y¼
sequential/activation/sub_1Subsequential/activation/mul_1:z:0&sequential/activation/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/sub_1
sequential/activation/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
sequential/activation/truediv/xÆ
sequential/activation/truedivRealDiv(sequential/activation/truediv/x:output:0sequential/activation/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/truediv
sequential/activation/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/activation/sub_2/x¾
sequential/activation/sub_2Sub&sequential/activation/sub_2/x:output:0!sequential/activation/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/sub_2
sequential/activation/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
sequential/activation/mul_2/x¾
sequential/activation/mul_2Mul&sequential/activation/mul_2/x:output:0!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/mul_2
sequential/activation/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/activation/add/y¸
sequential/activation/addAddV2sequential/activation/mul_2:z:0$sequential/activation/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/add
"sequential/activation/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential/activation/SelectV2_1/eõ
 sequential/activation/SelectV2_1SelectV2%sequential/activation/LessEqual_1:z:0sequential/activation/add:z:0+sequential/activation/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential/activation/SelectV2_1
"sequential/activation/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential/activation/SelectV2_2/eý
 sequential/activation/SelectV2_2SelectV2!sequential/activation/Greater:z:0)sequential/activation/SelectV2_1:output:0+sequential/activation/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential/activation/SelectV2_2
sequential/activation/LogLog)sequential/activation/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/Log
sequential/activation/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
sequential/activation/mul_3/xº
sequential/activation/mul_3Mul&sequential/activation/mul_3/x:output:0sequential/activation/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/mul_3
sequential/activation/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/activation/add_1/y¾
sequential/activation/add_1AddV2sequential/activation/mul_3:z:0&sequential/activation/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/add_1é
 sequential/activation/SelectV2_3SelectV2#sequential/activation/Greater_1:z:0sequential/activation/sub_2:z:0sequential/activation/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential/activation/SelectV2_3ñ
 sequential/activation/SelectV2_4SelectV2#sequential/activation/LessEqual:z:0sequential/activation/mul:z:0)sequential/activation/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential/activation/SelectV2_4Æ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpÏ
sequential/dense_1/MatMulMatMul)sequential/activation/SelectV2_4:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
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
sequential/dense_1/BiasAdd
sequential/activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/activation_1/ConstÒ
!sequential/activation_1/LessEqual	LessEqual#sequential/dense_1/BiasAdd:output:0&sequential/activation_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/activation_1/LessEqual
sequential/activation_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/activation_1/Const_1Î
sequential/activation_1/GreaterGreater#sequential/dense_1/BiasAdd:output:0(sequential/activation_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/activation_1/Greater
sequential/activation_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2!
sequential/activation_1/Const_2Ò
!sequential/activation_1/Greater_1Greater#sequential/dense_1/BiasAdd:output:0(sequential/activation_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/activation_1/Greater_1
sequential/activation_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2!
sequential/activation_1/Const_3Ø
#sequential/activation_1/LessEqual_1	LessEqual#sequential/dense_1/BiasAdd:output:0(sequential/activation_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential/activation_1/LessEqual_1
sequential/activation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/activation_1/sub/yÀ
sequential/activation_1/subSub#sequential/dense_1/BiasAdd:output:0&sequential/activation_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/sub
"sequential/activation_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential/activation_1/SelectV2/e÷
 sequential/activation_1/SelectV2SelectV2%sequential/activation_1/LessEqual:z:0sequential/activation_1/sub:z:0+sequential/activation_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential/activation_1/SelectV2
sequential/activation_1/ExpExp)sequential/activation_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/Exp
sequential/activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
sequential/activation_1/mul/x¼
sequential/activation_1/mulMul&sequential/activation_1/mul/x:output:0sequential/activation_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/mul
sequential/activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2!
sequential/activation_1/mul_1/xÆ
sequential/activation_1/mul_1Mul(sequential/activation_1/mul_1/x:output:0#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/mul_1
sequential/activation_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2!
sequential/activation_1/sub_1/yÄ
sequential/activation_1/sub_1Sub!sequential/activation_1/mul_1:z:0(sequential/activation_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/sub_1
!sequential/activation_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential/activation_1/truediv/xÎ
sequential/activation_1/truedivRealDiv*sequential/activation_1/truediv/x:output:0!sequential/activation_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/activation_1/truediv
sequential/activation_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
sequential/activation_1/sub_2/xÆ
sequential/activation_1/sub_2Sub(sequential/activation_1/sub_2/x:output:0#sequential/activation_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/sub_2
sequential/activation_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2!
sequential/activation_1/mul_2/xÆ
sequential/activation_1/mul_2Mul(sequential/activation_1/mul_2/x:output:0#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/mul_2
sequential/activation_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/activation_1/add/yÀ
sequential/activation_1/addAddV2!sequential/activation_1/mul_2:z:0&sequential/activation_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/add
$sequential/activation_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$sequential/activation_1/SelectV2_1/eÿ
"sequential/activation_1/SelectV2_1SelectV2'sequential/activation_1/LessEqual_1:z:0sequential/activation_1/add:z:0-sequential/activation_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential/activation_1/SelectV2_1
$sequential/activation_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$sequential/activation_1/SelectV2_2/e
"sequential/activation_1/SelectV2_2SelectV2#sequential/activation_1/Greater:z:0+sequential/activation_1/SelectV2_1:output:0-sequential/activation_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential/activation_1/SelectV2_2 
sequential/activation_1/LogLog+sequential/activation_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/Log
sequential/activation_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2!
sequential/activation_1/mul_3/xÂ
sequential/activation_1/mul_3Mul(sequential/activation_1/mul_3/x:output:0sequential/activation_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/mul_3
sequential/activation_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential/activation_1/add_1/yÆ
sequential/activation_1/add_1AddV2!sequential/activation_1/mul_3:z:0(sequential/activation_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation_1/add_1ó
"sequential/activation_1/SelectV2_3SelectV2%sequential/activation_1/Greater_1:z:0!sequential/activation_1/sub_2:z:0!sequential/activation_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential/activation_1/SelectV2_3û
"sequential/activation_1/SelectV2_4SelectV2%sequential/activation_1/LessEqual:z:0sequential/activation_1/mul:z:0+sequential/activation_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential/activation_1/SelectV2_4©
IdentityIdentity+sequential/activation_1/SelectV2_4:output:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
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
!
`
D__inference_activation_layer_call_and_return_conditional_losses_2413

inputs
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
	LessEqual	LessEqualinputsConst:output:0*
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
Const_1i
GreaterGreaterinputsConst_1:output:0*
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
Const_2m
	Greater_1GreaterinputsConst_2:output:0*
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
Const_3s
LessEqual_1	LessEqualinputsConst_3:output:0*
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
sub/y[
subSubinputssub/y:output:0*
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
mul_1/xa
mul_1Mulmul_1/x:output:0inputs*
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
mul_2/xa
mul_2Mulmul_2/x:output:0inputs*
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

SelectV2_4g
IdentityIdentitySelectV2_4:output:0*
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
Ë	
ð
?__inference_dense_layer_call_and_return_conditional_losses_3066

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
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
Ñ

D__inference_sequential_layer_call_and_return_conditional_losses_2550

inputs

dense_2537:

dense_2539:
dense_1_2543:
dense_1_2545:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallÿ
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_2537
dense_2539*
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
GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_23652
dense/StatefulPartitionedCallø
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_24132
activation/PartitionedCall¦
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_2543dense_1_2545*
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
GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24252!
dense_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_24732
activation_1/PartitionedCall»
IdentityIdentity%activation_1/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
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
Ë	
ð
?__inference_dense_layer_call_and_return_conditional_losses_2365

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
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
À
G
+__inference_activation_1_layer_call_fn_3137

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_24732
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
!
b
F__inference_activation_1_layer_call_and_return_conditional_losses_3179

inputs
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
	LessEqual	LessEqualinputsConst:output:0*
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
Const_1i
GreaterGreaterinputsConst_1:output:0*
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
Const_2m
	Greater_1GreaterinputsConst_2:output:0*
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
Const_3s
LessEqual_1	LessEqualinputsConst_3:output:0*
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
sub/y[
subSubinputssub/y:output:0*
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
mul_1/xa
mul_1Mulmul_1/x:output:0inputs*
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
mul_2/xa
mul_2Mulmul_2/x:output:0inputs*
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

SelectV2_4g
IdentityIdentitySelectV2_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
!
b
F__inference_activation_1_layer_call_and_return_conditional_losses_2473

inputs
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
	LessEqual	LessEqualinputsConst:output:0*
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
Const_1i
GreaterGreaterinputsConst_1:output:0*
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
Const_2m
	Greater_1GreaterinputsConst_2:output:0*
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
Const_3s
LessEqual_1	LessEqualinputsConst_3:output:0*
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
sub/y[
subSubinputssub/y:output:0*
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
mul_1/xa
mul_1Mulmul_1/x:output:0inputs*
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
mul_2/xa
mul_2Mulmul_2/x:output:0inputs*
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

SelectV2_4g
IdentityIdentitySelectV2_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²f
Ä
D__inference_sequential_layer_call_and_return_conditional_losses_2955
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
dense/BiasAddi
activation/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/Const
activation/LessEqual	LessEqualdense/BiasAdd:output:0activation/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/LessEqualm
activation/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/Const_1
activation/GreaterGreaterdense/BiasAdd:output:0activation/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Greaterm
activation/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation/Const_2
activation/Greater_1Greaterdense/BiasAdd:output:0activation/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Greater_1m
activation/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation/Const_3¤
activation/LessEqual_1	LessEqualdense/BiasAdd:output:0activation/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/LessEqual_1i
activation/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/sub/y
activation/subSubdense/BiasAdd:output:0activation/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/subs
activation/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2/e¶
activation/SelectV2SelectV2activation/LessEqual:z:0activation/sub:z:0activation/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2w
activation/ExpExpactivation/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Expi
activation/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
activation/mul/x
activation/mulMulactivation/mul/x:output:0activation/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mulm
activation/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
activation/mul_1/x
activation/mul_1Mulactivation/mul_1/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_1m
activation/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
activation/sub_1/y
activation/sub_1Subactivation/mul_1:z:0activation/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/sub_1q
activation/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/truediv/x
activation/truedivRealDivactivation/truediv/x:output:0activation/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/truedivm
activation/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/sub_2/x
activation/sub_2Subactivation/sub_2/x:output:0activation/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/sub_2m
activation/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
activation/mul_2/x
activation/mul_2Mulactivation/mul_2/x:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_2i
activation/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation/add/y
activation/addAddV2activation/mul_2:z:0activation/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/addw
activation/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2_1/e¾
activation/SelectV2_1SelectV2activation/LessEqual_1:z:0activation/add:z:0 activation/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_1w
activation/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation/SelectV2_2/eÆ
activation/SelectV2_2SelectV2activation/Greater:z:0activation/SelectV2_1:output:0 activation/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_2y
activation/LogLogactivation/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Logm
activation/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
activation/mul_3/x
activation/mul_3Mulactivation/mul_3/x:output:0activation/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/mul_3m
activation/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation/add_1/y
activation/add_1AddV2activation/mul_3:z:0activation/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/add_1²
activation/SelectV2_3SelectV2activation/Greater_1:z:0activation/sub_2:z:0activation/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_3º
activation/SelectV2_4SelectV2activation/LessEqual:z:0activation/mul:z:0activation/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/SelectV2_4¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp£
dense_1/MatMulMatMulactivation/SelectV2_4:output:0%dense_1/MatMul/ReadVariableOp:value:0*
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
dense_1/BiasAddm
activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/Const¦
activation_1/LessEqual	LessEqualdense_1/BiasAdd:output:0activation_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/LessEqualq
activation_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/Const_1¢
activation_1/GreaterGreaterdense_1/BiasAdd:output:0activation_1/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Greaterq
activation_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation_1/Const_2¦
activation_1/Greater_1Greaterdense_1/BiasAdd:output:0activation_1/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Greater_1q
activation_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  pA2
activation_1/Const_3¬
activation_1/LessEqual_1	LessEqualdense_1/BiasAdd:output:0activation_1/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/LessEqual_1m
activation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/sub/y
activation_1/subSubdense_1/BiasAdd:output:0activation_1/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/subw
activation_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2/eÀ
activation_1/SelectV2SelectV2activation_1/LessEqual:z:0activation_1/sub:z:0 activation_1/SelectV2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2}
activation_1/ExpExpactivation_1/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Expm
activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Tø­?2
activation_1/mul/x
activation_1/mulMulactivation_1/mul/x:output:0activation_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mulq
activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ð+ÚB2
activation_1/mul_1/x
activation_1/mul_1Mulactivation_1/mul_1/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_1q
activation_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *k¯D2
activation_1/sub_1/y
activation_1/sub_1Subactivation_1/mul_1:z:0activation_1/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/sub_1u
activation_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/truediv/x¢
activation_1/truedivRealDivactivation_1/truediv/x:output:0activation_1/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/truedivq
activation_1/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/sub_2/x
activation_1/sub_2Subactivation_1/sub_2/x:output:0activation_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/sub_2q
activation_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
activation_1/mul_2/x
activation_1/mul_2Mulactivation_1/mul_2/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_2m
activation_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/add/y
activation_1/addAddV2activation_1/mul_2:z:0activation_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/add{
activation_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2_1/eÈ
activation_1/SelectV2_1SelectV2activation_1/LessEqual_1:z:0activation_1/add:z:0"activation_1/SelectV2_1/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_1{
activation_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_1/SelectV2_2/eÐ
activation_1/SelectV2_2SelectV2activation_1/Greater:z:0 activation_1/SelectV2_1:output:0"activation_1/SelectV2_2/e:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_2
activation_1/LogLog activation_1/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/Logq
activation_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *Âõ<2
activation_1/mul_3/x
activation_1/mul_3Mulactivation_1/mul_3/x:output:0activation_1/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/mul_3q
activation_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_1/add_1/y
activation_1/add_1AddV2activation_1/mul_3:z:0activation_1/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/add_1¼
activation_1/SelectV2_3SelectV2activation_1/Greater_1:z:0activation_1/sub_2:z:0activation_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_3Ä
activation_1/SelectV2_4SelectV2activation_1/LessEqual:z:0activation_1/mul:z:0 activation_1/SelectV2_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_1/SelectV2_4ò
IdentityIdentity activation_1/SelectV2_4:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
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
Í	
ò
A__inference_dense_1_layer_call_and_return_conditional_losses_2425

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
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_default£
C
dense_input4
serving_default_dense_input:0ÿÿÿÿÿÿÿÿÿ@
activation_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
Ï"
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
K__call__
*L&call_and_return_all_conditional_losses
M_default_save_signature" 
_tf_keras_sequentialý{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "custom_activation"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_activation"}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [256, 2]}, "float32", "dense_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "custom_activation"}, "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_activation"}, "shared_object_id": 8}]}}, "training_config": {"loss": {"class_name": "MeanAbsoluteError", "config": {"reduction": "auto", "name": "mean_absolute_error"}, "shared_object_id": 11}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
Ç

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
N__call__
*O&call_and_return_all_conditional_losses"¢
_tf_keras_layer{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [256, 2]}}
õ
regularization_losses
trainable_variables
	variables
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"æ
_tf_keras_layerÌ{"name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "custom_activation"}, "shared_object_id": 4}
Ë

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"¦
_tf_keras_layer{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [256, 2]}}
ù
regularization_losses
trainable_variables
	variables
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_activation"}, "shared_object_id": 8}
¯
iter

 beta_1

!beta_2
	"decay
#learning_rate
$momentum_cachemCmDmEmFvGvHvIvJ"
	optimizer
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Ê
regularization_losses

%layers
&layer_regularization_losses
trainable_variables
'layer_metrics
	variables
(metrics
)non_trainable_variables
K__call__
M_default_save_signature
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
,
Vserving_default"
signature_map
:2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses

*layers
+layer_regularization_losses
trainable_variables
,layer_metrics
	variables
-metrics
.non_trainable_variables
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses

/layers
0layer_regularization_losses
trainable_variables
1layer_metrics
	variables
2metrics
3non_trainable_variables
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses

4layers
5layer_regularization_losses
trainable_variables
6layer_metrics
	variables
7metrics
8non_trainable_variables
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses

9layers
:layer_regularization_losses
trainable_variables
;layer_metrics
	variables
<metrics
=non_trainable_variables
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
>0"
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
Ô
	?total
	@count
A	variables
B	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 13}
:  (2total
:  (2count
.
?0
@1"
trackable_list_wrapper
-
A	variables"
_generic_user_object
$:"2Nadam/dense/kernel/m
:2Nadam/dense/bias/m
&:$2Nadam/dense_1/kernel/m
 :2Nadam/dense_1/bias/m
$:"2Nadam/dense/kernel/v
:2Nadam/dense/bias/v
&:$2Nadam/dense_1/kernel/v
 :2Nadam/dense_1/bias/v
ò2ï
)__inference_sequential_layer_call_fn_2640
)__inference_sequential_layer_call_fn_2653
)__inference_sequential_layer_call_fn_2666
)__inference_sequential_layer_call_fn_2679À
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
Þ2Û
D__inference_sequential_layer_call_and_return_conditional_losses_2771
D__inference_sequential_layer_call_and_return_conditional_losses_2863
D__inference_sequential_layer_call_and_return_conditional_losses_2955
D__inference_sequential_layer_call_and_return_conditional_losses_3047À
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
á2Þ
__inference__wrapped_model_2348º
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
Î2Ë
$__inference_dense_layer_call_fn_3056¢
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
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_3066¢
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
Ó2Ð
)__inference_activation_layer_call_fn_3071¢
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
î2ë
D__inference_activation_layer_call_and_return_conditional_losses_3113¢
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
Ð2Í
&__inference_dense_1_layer_call_fn_3122¢
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
A__inference_dense_1_layer_call_and_return_conditional_losses_3132¢
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
Õ2Ò
+__inference_activation_1_layer_call_fn_3137¢
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
ð2í
F__inference_activation_1_layer_call_and_return_conditional_losses_3179¢
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
ÍBÊ
"__inference_signature_wrapper_2627dense_input"
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
 
__inference__wrapped_model_2348y4¢1
*¢'
%"
dense_inputÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
activation_1&#
activation_1ÿÿÿÿÿÿÿÿÿ¢
F__inference_activation_1_layer_call_and_return_conditional_losses_3179X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
+__inference_activation_1_layer_call_fn_3137K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ 
D__inference_activation_layer_call_and_return_conditional_losses_3113X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
)__inference_activation_layer_call_fn_3071K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_1_layer_call_and_return_conditional_losses_3132\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_1_layer_call_fn_3122O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
?__inference_dense_layer_call_and_return_conditional_losses_3066\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
$__inference_dense_layer_call_fn_3056O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
D__inference_sequential_layer_call_and_return_conditional_losses_2771f7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ®
D__inference_sequential_layer_call_and_return_conditional_losses_2863f7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ³
D__inference_sequential_layer_call_and_return_conditional_losses_2955k<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ³
D__inference_sequential_layer_call_and_return_conditional_losses_3047k<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_sequential_layer_call_fn_2640^<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_2653Y7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_2666Y7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_2679^<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¯
"__inference_signature_wrapper_2627C¢@
¢ 
9ª6
4
dense_input%"
dense_inputÿÿÿÿÿÿÿÿÿ";ª8
6
activation_1&#
activation_1ÿÿÿÿÿÿÿÿÿ
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE eagle SYSTEM "eagle.dtd">
<eagle version="9.6.2">
<drawing>
<settings>
<setting alwaysvectorfont="no"/>
<setting verticaltext="up"/>
</settings>
<grid distance="0.1" unitdist="inch" unit="inch" style="lines" multiple="1" display="no" altdistance="0.01" altunitdist="inch" altunit="inch"/>
<layers>
<layer number="1" name="Top" color="4" fill="1" visible="no" active="no"/>
<layer number="16" name="Bottom" color="1" fill="1" visible="no" active="no"/>
<layer number="17" name="Pads" color="2" fill="1" visible="no" active="no"/>
<layer number="18" name="Vias" color="2" fill="1" visible="no" active="no"/>
<layer number="19" name="Unrouted" color="6" fill="1" visible="no" active="no"/>
<layer number="20" name="Dimension" color="24" fill="1" visible="no" active="no"/>
<layer number="21" name="tPlace" color="7" fill="1" visible="no" active="no"/>
<layer number="22" name="bPlace" color="7" fill="1" visible="no" active="no"/>
<layer number="23" name="tOrigins" color="15" fill="1" visible="no" active="no"/>
<layer number="24" name="bOrigins" color="15" fill="1" visible="no" active="no"/>
<layer number="25" name="tNames" color="7" fill="1" visible="no" active="no"/>
<layer number="26" name="bNames" color="7" fill="1" visible="no" active="no"/>
<layer number="27" name="tValues" color="7" fill="1" visible="no" active="no"/>
<layer number="28" name="bValues" color="7" fill="1" visible="no" active="no"/>
<layer number="29" name="tStop" color="7" fill="3" visible="no" active="no"/>
<layer number="30" name="bStop" color="7" fill="6" visible="no" active="no"/>
<layer number="31" name="tCream" color="7" fill="4" visible="no" active="no"/>
<layer number="32" name="bCream" color="7" fill="5" visible="no" active="no"/>
<layer number="33" name="tFinish" color="6" fill="3" visible="no" active="no"/>
<layer number="34" name="bFinish" color="6" fill="6" visible="no" active="no"/>
<layer number="35" name="tGlue" color="7" fill="4" visible="no" active="no"/>
<layer number="36" name="bGlue" color="7" fill="5" visible="no" active="no"/>
<layer number="37" name="tTest" color="7" fill="1" visible="no" active="no"/>
<layer number="38" name="bTest" color="7" fill="1" visible="no" active="no"/>
<layer number="39" name="tKeepout" color="4" fill="11" visible="no" active="no"/>
<layer number="40" name="bKeepout" color="1" fill="11" visible="no" active="no"/>
<layer number="41" name="tRestrict" color="4" fill="10" visible="no" active="no"/>
<layer number="42" name="bRestrict" color="1" fill="10" visible="no" active="no"/>
<layer number="43" name="vRestrict" color="2" fill="10" visible="no" active="no"/>
<layer number="44" name="Drills" color="7" fill="1" visible="no" active="no"/>
<layer number="45" name="Holes" color="7" fill="1" visible="no" active="no"/>
<layer number="46" name="Milling" color="3" fill="1" visible="no" active="no"/>
<layer number="47" name="Measures" color="7" fill="1" visible="no" active="no"/>
<layer number="48" name="Document" color="7" fill="1" visible="no" active="no"/>
<layer number="49" name="Reference" color="7" fill="1" visible="no" active="no"/>
<layer number="51" name="tDocu" color="7" fill="1" visible="no" active="no"/>
<layer number="52" name="bDocu" color="7" fill="1" visible="no" active="no"/>
<layer number="88" name="SimResults" color="9" fill="1" visible="yes" active="yes"/>
<layer number="89" name="SimProbes" color="9" fill="1" visible="yes" active="yes"/>
<layer number="90" name="Modules" color="5" fill="1" visible="yes" active="yes"/>
<layer number="91" name="Nets" color="2" fill="1" visible="yes" active="yes"/>
<layer number="92" name="Busses" color="1" fill="1" visible="yes" active="yes"/>
<layer number="93" name="Pins" color="2" fill="1" visible="no" active="yes"/>
<layer number="94" name="Symbols" color="4" fill="1" visible="yes" active="yes"/>
<layer number="95" name="Names" color="7" fill="1" visible="yes" active="yes"/>
<layer number="96" name="Values" color="7" fill="1" visible="yes" active="yes"/>
<layer number="97" name="Info" color="7" fill="1" visible="yes" active="yes"/>
<layer number="98" name="Guide" color="6" fill="1" visible="yes" active="yes"/>
</layers>
<schematic xreflabel="%F%N/%S.%C%R" xrefpart="/%S.%C%R">
<libraries>
<library name="Library">
<packages>
<package name="DIP5918W53P254L2279H300Q12B" urn="urn:adsk.eagle:footprint:38566975/1" locally_modified="yes">
<description>12-DIP, 2.54 mm (0.10 in) pitch, 59.18 mm (2.33 in) span, 22.80 X 34.41 X 3.00 mm body
&lt;p&gt;12-pin DIP package with 2.54 mm (0.10 in) pitch, 59.18 mm (2.33 in) span with body size 22.80 X 34.41 X 3.00 mm&lt;/p&gt;</description>
<circle x="-31.619" y="6.35" radius="0.25" width="0" layer="21"/>
<wire x1="-31.115" y1="7.2757" x2="-31.115" y2="12.95" width="0.12" layer="21"/>
<wire x1="-31.115" y1="12.95" x2="31.115" y2="12.95" width="0.12" layer="21"/>
<wire x1="31.115" y1="12.95" x2="31.115" y2="7.2757" width="0.12" layer="21"/>
<wire x1="-31.115" y1="-7.2757" x2="-31.115" y2="-12.95" width="0.12" layer="21"/>
<wire x1="-31.115" y1="-12.95" x2="31.115" y2="-12.95" width="0.12" layer="21"/>
<wire x1="31.115" y1="-12.95" x2="31.115" y2="-7.2757" width="0.12" layer="21"/>
<wire x1="31.115" y1="-12.95" x2="-31.115" y2="-12.95" width="0.12" layer="51"/>
<wire x1="-31.115" y1="-12.95" x2="-31.115" y2="12.95" width="0.12" layer="51"/>
<wire x1="-31.115" y1="12.95" x2="31.115" y2="12.95" width="0.12" layer="51"/>
<wire x1="31.115" y1="12.95" x2="31.115" y2="-12.95" width="0.12" layer="51"/>
<pad name="GND01" x="-29.591" y="6.35" drill="0.7434" diameter="1.3434"/>
<pad name="OE01" x="-29.591" y="3.81" drill="0.7434" diameter="1.3434"/>
<pad name="SCL01" x="-29.591" y="1.27" drill="0.7434" diameter="1.3434"/>
<pad name="SDA01" x="-29.591" y="-1.27" drill="0.7434" diameter="1.3434"/>
<pad name="VCC01" x="-29.591" y="-3.81" drill="0.7434" diameter="1.3434"/>
<pad name="V+01" x="-29.591" y="-6.35" drill="0.7434" diameter="1.3434"/>
<pad name="V+02" x="29.591" y="-6.35" drill="0.7434" diameter="1.3434"/>
<pad name="VCC02" x="29.591" y="-3.81" drill="0.7434" diameter="1.3434"/>
<pad name="SDA02" x="29.591" y="-1.27" drill="0.7434" diameter="1.3434"/>
<pad name="SCL02" x="29.591" y="1.27" drill="0.7434" diameter="1.3434"/>
<pad name="OE02" x="29.591" y="3.81" drill="0.7434" diameter="1.3434"/>
<pad name="GND02" x="29.591" y="6.35" drill="0.7434" diameter="1.3434"/>
<text x="0" y="13.585" size="1.27" layer="25" align="bottom-center">&gt;NAME</text>
<text x="0" y="-13.585" size="1.27" layer="27" align="top-center">&gt;VALUE</text>
<pad name="PWM1" x="-24.13" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC1" x="-24.13" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND1" x="-24.13" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM2" x="-21.59" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC2" x="-21.59" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND2" x="-21.59" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM3" x="-19.05" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC3" x="-19.05" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND3" x="-19.05" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM4" x="-16.51" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC4" x="-16.51" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND4" x="-16.51" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM5" x="-11.43" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC5" x="-11.43" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND5" x="-11.43" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM6" x="-8.89" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC6" x="-8.89" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND6" x="-8.89" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM7" x="-6.35" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC7" x="-6.35" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND7" x="-6.35" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM8" x="-3.81" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC8" x="-3.81" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND8" x="-3.81" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM9" x="3.81" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC9" x="3.81" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND9" x="3.81" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM10" x="6.35" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC10" x="6.35" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND10" x="6.35" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM11" x="8.89" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC11" x="8.89" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND11" x="8.89" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM12" x="11.43" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC12" x="11.43" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND12" x="11.43" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM13" x="16.51" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC13" x="16.51" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND13" x="16.51" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM14" x="19.05" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC14" x="19.05" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND14" x="19.05" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM15" x="21.59" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC15" x="21.59" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND15" x="21.59" y="-10.16" drill="0.6" shape="square"/>
<pad name="PWM16" x="24.13" y="-5.08" drill="0.6" shape="square"/>
<pad name="VCC16" x="24.13" y="-7.62" drill="0.6" shape="square"/>
<pad name="GND16" x="24.13" y="-10.16" drill="0.6" shape="square"/>
<pad name="GND03" x="-1.27" y="11.43" drill="0.6" shape="square"/>
<pad name="V+03" x="1.27" y="11.43" drill="0.6" shape="square"/>
</package>
</packages>
<packages3d>
<package3d name="DIP5918W53P254L2279H300Q12B" urn="urn:adsk.eagle:package:38566682/2" locally_modified="yes" type="model">
<description>12-DIP, 2.54 mm (0.10 in) pitch, 59.18 mm (2.33 in) span, 22.80 X 34.41 X 3.00 mm body
&lt;p&gt;12-pin DIP package with 2.54 mm (0.10 in) pitch, 59.18 mm (2.33 in) span with body size 22.80 X 34.41 X 3.00 mm&lt;/p&gt;</description>
<packageinstances>
<packageinstance name="DIP5918W53P254L2279H300Q12B"/>
</packageinstances>
</package3d>
</packages3d>
<symbols>
<symbol name="PCA9685">
<pin name="GND01" x="-53.34" y="30.48" length="middle"/>
<pin name="OE01" x="-53.34" y="25.4" length="middle"/>
<pin name="SCL01" x="-53.34" y="20.32" length="middle"/>
<pin name="SDA01" x="-53.34" y="15.24" length="middle"/>
<pin name="VCC01" x="-53.34" y="10.16" length="middle"/>
<pin name="V+01" x="-53.34" y="5.08" length="middle"/>
<pin name="GND02" x="83.82" y="30.48" length="middle" rot="R180"/>
<pin name="OE02" x="83.82" y="25.4" length="middle" rot="R180"/>
<pin name="SCL02" x="83.82" y="20.32" length="middle" rot="R180"/>
<pin name="SDA02" x="83.82" y="15.24" length="middle" rot="R180"/>
<pin name="VCC02" x="83.82" y="10.16" length="middle" rot="R180"/>
<pin name="V+02" x="83.82" y="5.08" length="middle" rot="R180"/>
<pin name="PWM$1" x="-33.02" y="30.48" visible="pad" length="middle"/>
<pin name="VCC$1" x="-33.02" y="25.4" visible="pad" length="middle"/>
<pin name="GND$1" x="-33.02" y="20.32" visible="pad" length="middle"/>
<pin name="PWM$2" x="-15.24" y="30.48" visible="pad" length="middle"/>
<pin name="VCC$2" x="-15.24" y="25.4" visible="pad" length="middle"/>
<pin name="GND$2" x="-15.24" y="20.32" visible="pad" length="middle"/>
<pin name="PWM$3" x="2.54" y="30.48" visible="pad" length="middle"/>
<pin name="VCC$3" x="2.54" y="25.4" visible="pad" length="middle"/>
<pin name="GND$3" x="2.54" y="20.32" visible="pad" length="middle"/>
<pin name="PWM$4" x="17.78" y="30.48" visible="pad" length="middle"/>
<pin name="VCC$4" x="17.78" y="25.4" visible="pad" length="middle"/>
<pin name="GND$4" x="17.78" y="20.32" visible="pad" length="middle"/>
<pin name="PWM$5" x="35.56" y="30.48" visible="pad" length="middle"/>
<pin name="VCC$5" x="35.56" y="25.4" visible="pad" length="middle"/>
<pin name="GND$5" x="35.56" y="20.32" visible="pad" length="middle"/>
<pin name="PWM$6" x="53.34" y="30.48" visible="pad" length="middle"/>
<pin name="VCC$6" x="53.34" y="25.4" visible="pad" length="middle"/>
<pin name="GND$6" x="53.34" y="20.32" visible="pad" length="middle"/>
<wire x1="-53.34" y1="33.02" x2="-53.34" y2="2.54" width="0.254" layer="94"/>
<wire x1="-53.34" y1="2.54" x2="83.82" y2="2.54" width="0.254" layer="94"/>
<wire x1="83.82" y1="2.54" x2="83.82" y2="33.02" width="0.254" layer="94"/>
<wire x1="83.82" y1="33.02" x2="-53.34" y2="33.02" width="0.254" layer="94"/>
<pin name="GND03" x="17.78" y="2.54" length="middle" rot="R90"/>
<pin name="V+03" x="12.7" y="2.54" length="middle" rot="R90"/>
</symbol>
</symbols>
<devicesets>
<deviceset name="PCA9685">
<gates>
<gate name="G$1" symbol="PCA9685" x="-17.78" y="-15.24"/>
</gates>
<devices>
<device name="PCA9685" package="DIP5918W53P254L2279H300Q12B">
<connects>
<connect gate="G$1" pin="GND$1" pad="GND1"/>
<connect gate="G$1" pin="GND$2" pad="GND2"/>
<connect gate="G$1" pin="GND$3" pad="GND3"/>
<connect gate="G$1" pin="GND$4" pad="GND4"/>
<connect gate="G$1" pin="GND$5" pad="GND5"/>
<connect gate="G$1" pin="GND$6" pad="GND6"/>
<connect gate="G$1" pin="GND01" pad="GND01"/>
<connect gate="G$1" pin="GND02" pad="GND02"/>
<connect gate="G$1" pin="GND03" pad="GND03"/>
<connect gate="G$1" pin="OE01" pad="OE01"/>
<connect gate="G$1" pin="OE02" pad="OE02"/>
<connect gate="G$1" pin="PWM$1" pad="PWM1"/>
<connect gate="G$1" pin="PWM$2" pad="PWM2"/>
<connect gate="G$1" pin="PWM$3" pad="PWM3"/>
<connect gate="G$1" pin="PWM$4" pad="PWM4"/>
<connect gate="G$1" pin="PWM$5" pad="PWM5"/>
<connect gate="G$1" pin="PWM$6" pad="PWM6"/>
<connect gate="G$1" pin="SCL01" pad="SCL01"/>
<connect gate="G$1" pin="SCL02" pad="SCL02"/>
<connect gate="G$1" pin="SDA01" pad="SDA01"/>
<connect gate="G$1" pin="SDA02" pad="SDA02"/>
<connect gate="G$1" pin="V+01" pad="V+01"/>
<connect gate="G$1" pin="V+02" pad="V+02"/>
<connect gate="G$1" pin="V+03" pad="V+03"/>
<connect gate="G$1" pin="VCC$1" pad="VCC1"/>
<connect gate="G$1" pin="VCC$2" pad="VCC2"/>
<connect gate="G$1" pin="VCC$3" pad="VCC3"/>
<connect gate="G$1" pin="VCC$4" pad="VCC4"/>
<connect gate="G$1" pin="VCC$5" pad="VCC5"/>
<connect gate="G$1" pin="VCC$6" pad="VCC6"/>
<connect gate="G$1" pin="VCC01" pad="VCC01"/>
<connect gate="G$1" pin="VCC02" pad="VCC02"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:38566682/2"/>
</package3dinstances>
<technologies>
<technology name=""/>
</technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
<library name="dc-dc-converter" urn="urn:adsk.eagle:library:208">
<description>&lt;b&gt;DC-DC Converters&lt;/b&gt;&lt;p&gt;
&lt;author&gt;Created by librarian@cadsoft.de&lt;/author&gt;</description>
<packages>
<package name="NME" urn="urn:adsk.eagle:footprint:12288/1" library_version="2">
<description>&lt;b&gt;DC-DC CONVERTER&lt;/b&gt;</description>
<wire x1="-1.143" y1="5.842" x2="4.953" y2="5.842" width="0.1524" layer="21"/>
<wire x1="4.953" y1="5.842" x2="4.953" y2="-5.842" width="0.1524" layer="21"/>
<wire x1="4.953" y1="-5.842" x2="-1.143" y2="-5.842" width="0.1524" layer="21"/>
<wire x1="-1.143" y1="-5.842" x2="-1.143" y2="-4.572" width="0.1524" layer="21"/>
<wire x1="-1.143" y1="-4.572" x2="-1.143" y2="-3.048" width="0.1524" layer="51"/>
<wire x1="-1.143" y1="4.572" x2="-1.143" y2="5.842" width="0.1524" layer="21"/>
<wire x1="-1.143" y1="-2.032" x2="-1.143" y2="-0.508" width="0.1524" layer="51"/>
<wire x1="-1.143" y1="-3.048" x2="-1.143" y2="-2.032" width="0.1524" layer="21"/>
<wire x1="-1.143" y1="-0.508" x2="-1.143" y2="0.508" width="0.1524" layer="21"/>
<wire x1="-1.143" y1="0.508" x2="-1.143" y2="2.032" width="0.1524" layer="51"/>
<wire x1="-1.143" y1="2.032" x2="-1.143" y2="3.048" width="0.1524" layer="21"/>
<wire x1="-1.143" y1="3.048" x2="-1.143" y2="4.572" width="0.1524" layer="51"/>
<circle x="0" y="5.08" radius="0.254" width="0.1524" layer="21"/>
<pad name="1" x="0" y="3.81" drill="0.8128" shape="long"/>
<pad name="2" x="0" y="1.27" drill="0.8128" shape="long"/>
<pad name="3" x="0" y="-1.27" drill="0.8128" shape="long"/>
<pad name="4" x="0" y="-3.81" drill="0.8128" shape="long"/>
<text x="-1.143" y="6.223" size="1.27" layer="25" ratio="10">&gt;NAME</text>
<text x="3.048" y="-5.08" size="1.27" layer="27" ratio="10" rot="R90">&gt;VALUE</text>
<text x="4.572" y="-5.08" size="1.016" layer="21" ratio="12" rot="R90">DC-DC</text>
</package>
</packages>
<packages3d>
<package3d name="NME" urn="urn:adsk.eagle:package:12313/1" type="box" library_version="2">
<description>DC-DC CONVERTER</description>
<packageinstances>
<packageinstance name="NME"/>
</packageinstances>
</package3d>
</packages3d>
<symbols>
<symbol name="DC+" urn="urn:adsk.eagle:symbol:12287/1" library_version="2">
<wire x1="-10.16" y1="5.08" x2="10.16" y2="5.08" width="0.4064" layer="94"/>
<wire x1="10.16" y1="5.08" x2="10.16" y2="-7.62" width="0.4064" layer="94"/>
<wire x1="10.16" y1="-7.62" x2="-10.16" y2="-7.62" width="0.4064" layer="94"/>
<wire x1="-10.16" y1="-7.62" x2="-10.16" y2="5.08" width="0.4064" layer="94"/>
<text x="-10.16" y="5.715" size="1.778" layer="95">&gt;NAME</text>
<text x="-10.16" y="-10.16" size="1.778" layer="96">&gt;VALUE</text>
<text x="-8.636" y="-6.731" size="1.27" layer="94">DC/DC CONVERTER</text>
<pin name="+VIN" x="-12.7" y="2.54" length="short" direction="pas"/>
<pin name="-VIN" x="-12.7" y="-2.54" length="short" direction="pas"/>
<pin name="+VOUT" x="12.7" y="2.54" length="short" direction="pas" rot="R180"/>
<pin name="-VOUT" x="12.7" y="-2.54" length="short" direction="pas" rot="R180"/>
</symbol>
</symbols>
<devicesets>
<deviceset name="NME" urn="urn:adsk.eagle:component:12326/2" prefix="DC" uservalue="yes" library_version="2">
<description>&lt;b&gt;DC-DC CONVERTER&lt;/b&gt;</description>
<gates>
<gate name="G$1" symbol="DC+" x="0" y="0"/>
</gates>
<devices>
<device name="" package="NME">
<connects>
<connect gate="G$1" pin="+VIN" pad="2"/>
<connect gate="G$1" pin="+VOUT" pad="4"/>
<connect gate="G$1" pin="-VIN" pad="1"/>
<connect gate="G$1" pin="-VOUT" pad="3"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:12313/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="MF" value="C &amp; D TECHNOLOGIES, INC" constant="no"/>
<attribute name="MPN" value="NME1205D" constant="no"/>
<attribute name="OC_FARNELL" value="1021466" constant="no"/>
<attribute name="OC_NEWARK" value="98B8198" constant="no"/>
<attribute name="POPULARITY" value="2" constant="no"/>
</technology>
</technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
<library name="holes" urn="urn:adsk.eagle:library:237">
<description>&lt;b&gt;Mounting Holes and Pads&lt;/b&gt;&lt;p&gt;
&lt;author&gt;Created by librarian@cadsoft.de&lt;/author&gt;</description>
<packages>
<package name="2,8-PAD" urn="urn:adsk.eagle:footprint:14250/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 2.8 mm, round</description>
<wire x1="0" y1="2.921" x2="0" y2="2.667" width="0.0508" layer="21"/>
<wire x1="0" y1="-2.667" x2="0" y2="-2.921" width="0.0508" layer="21"/>
<wire x1="-1.778" y1="0" x2="0" y2="-1.778" width="2.286" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="1.778" x2="1.778" y2="0" width="2.286" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="0.635" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="2.921" width="0.1524" layer="21"/>
<circle x="0" y="0" radius="3.175" width="0.8128" layer="39"/>
<circle x="0" y="0" radius="3.175" width="0.8128" layer="40"/>
<circle x="0" y="0" radius="3.175" width="0.8128" layer="43"/>
<circle x="0" y="0" radius="1.5" width="0.2032" layer="21"/>
<pad name="B2,8" x="0" y="0" drill="2.8" diameter="5.334"/>
</package>
<package name="3,0-PAD" urn="urn:adsk.eagle:footprint:14251/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 3.0 mm, round</description>
<wire x1="-2.159" y1="0" x2="0" y2="-2.159" width="2.4892" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.159" x2="2.159" y2="0" width="2.4892" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="3.429" width="0.1524" layer="21"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="3.556" width="1.016" layer="39"/>
<circle x="0" y="0" radius="3.556" width="1.016" layer="40"/>
<circle x="0" y="0" radius="3.556" width="1.016" layer="43"/>
<circle x="0" y="0" radius="1.6" width="0.2032" layer="21"/>
<pad name="B3,0" x="0" y="0" drill="3" diameter="5.842"/>
<text x="-1.27" y="-3.81" size="1.27" layer="48">3,0</text>
</package>
<package name="3,2-PAD" urn="urn:adsk.eagle:footprint:14252/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 3.2 mm, round</description>
<wire x1="-2.159" y1="0" x2="0" y2="-2.159" width="2.4892" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.159" x2="2.159" y2="0" width="2.4892" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="3.429" width="0.1524" layer="21"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="3.683" width="1.27" layer="39"/>
<circle x="0" y="0" radius="3.683" width="1.27" layer="40"/>
<circle x="0" y="0" radius="3.556" width="1.016" layer="43"/>
<circle x="0" y="0" radius="1.7" width="0.1524" layer="21"/>
<pad name="B3,2" x="0" y="0" drill="3.2" diameter="5.842"/>
<text x="-1.27" y="-3.81" size="1.27" layer="48">3,2</text>
</package>
<package name="3,3-PAD" urn="urn:adsk.eagle:footprint:14253/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 3.3 mm, round</description>
<wire x1="-2.159" y1="0" x2="0" y2="-2.159" width="2.4892" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.159" x2="2.159" y2="0" width="2.4892" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="3.429" width="0.1524" layer="21"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="3.683" width="1.27" layer="39"/>
<circle x="0" y="0" radius="3.683" width="1.27" layer="40"/>
<circle x="0" y="0" radius="3.556" width="1.016" layer="43"/>
<circle x="0" y="0" radius="1.7" width="0.2032" layer="21"/>
<pad name="B3,3" x="0" y="0" drill="3.3" diameter="5.842"/>
</package>
<package name="3,6-PAD" urn="urn:adsk.eagle:footprint:14254/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 3.6 mm, round</description>
<wire x1="-2.159" y1="0" x2="0" y2="-2.159" width="2.4892" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.159" x2="2.159" y2="0" width="2.4892" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="3.429" width="0.1524" layer="21"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="3.683" width="1.397" layer="39"/>
<circle x="0" y="0" radius="3.683" width="1.397" layer="40"/>
<circle x="0" y="0" radius="3.556" width="1.016" layer="43"/>
<circle x="0" y="0" radius="1.9" width="0.2032" layer="21"/>
<pad name="B3,6" x="0" y="0" drill="3.6" diameter="5.842"/>
</package>
<package name="4,1-PAD" urn="urn:adsk.eagle:footprint:14255/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 4.1 mm, round</description>
<wire x1="-2.54" y1="0" x2="0" y2="-2.54" width="3.9116" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.54" x2="2.54" y2="0" width="3.9116" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="4.572" width="0.1524" layer="21"/>
<circle x="0" y="0" radius="5.08" width="2" layer="43"/>
<circle x="0" y="0" radius="2.15" width="0.2032" layer="21"/>
<pad name="B4,1" x="0" y="0" drill="4.1" diameter="8"/>
</package>
<package name="4,3-PAD" urn="urn:adsk.eagle:footprint:14256/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 4.3 mm, round</description>
<wire x1="-2.54" y1="0" x2="0" y2="-2.54" width="3.9116" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.54" x2="2.54" y2="0" width="3.9116" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="4.4958" width="0.1524" layer="51"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="5.588" width="2" layer="43"/>
<circle x="0" y="0" radius="5.588" width="2" layer="39"/>
<circle x="0" y="0" radius="5.588" width="2" layer="40"/>
<circle x="0" y="0" radius="2.25" width="0.1524" layer="21"/>
<pad name="B4,3" x="0" y="0" drill="4.3" diameter="9"/>
</package>
<package name="4,5-PAD" urn="urn:adsk.eagle:footprint:14257/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 4.5 mm, round</description>
<wire x1="-2.54" y1="0" x2="0" y2="-2.54" width="3.9116" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.54" x2="2.54" y2="0" width="3.9116" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="4.4958" width="0.1524" layer="51"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="5.588" width="2" layer="43"/>
<circle x="0" y="0" radius="5.588" width="2" layer="39"/>
<circle x="0" y="0" radius="5.588" width="2" layer="40"/>
<circle x="0" y="0" radius="2.35" width="0.1524" layer="21"/>
<pad name="B4,5" x="0" y="0" drill="4.5" diameter="9"/>
</package>
<package name="5,0-PAD" urn="urn:adsk.eagle:footprint:14258/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 5.0 mm, round</description>
<wire x1="-2.54" y1="0" x2="0" y2="-2.54" width="3.9116" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.54" x2="2.54" y2="0" width="3.9116" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="4.4958" width="0.1524" layer="51"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="5.588" width="2" layer="43"/>
<circle x="0" y="0" radius="5.588" width="2" layer="39"/>
<circle x="0" y="0" radius="5.588" width="2" layer="40"/>
<circle x="0" y="0" radius="2.6" width="0.1524" layer="21"/>
<pad name="B5" x="0" y="0" drill="5" diameter="9"/>
</package>
<package name="5,5-PAD" urn="urn:adsk.eagle:footprint:14259/1" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt; 5.5 mm, round</description>
<wire x1="-2.54" y1="0" x2="0" y2="-2.54" width="3.9116" layer="51" curve="90" cap="flat"/>
<wire x1="0" y1="2.54" x2="2.54" y2="0" width="3.9116" layer="51" curve="-90" cap="flat"/>
<circle x="0" y="0" radius="4.4958" width="0.1524" layer="51"/>
<circle x="0" y="0" radius="0.762" width="0.4572" layer="51"/>
<circle x="0" y="0" radius="5.588" width="2" layer="43"/>
<circle x="0" y="0" radius="5.588" width="2" layer="39"/>
<circle x="0" y="0" radius="5.588" width="2" layer="40"/>
<circle x="0" y="0" radius="2.85" width="0.1524" layer="21"/>
<pad name="B5,5" x="0" y="0" drill="5.5" diameter="9"/>
</package>
</packages>
<packages3d>
<package3d name="2,8-PAD" urn="urn:adsk.eagle:package:14281/1" type="box" library_version="2">
<description>MOUNTING PAD 2.8 mm, round</description>
<packageinstances>
<packageinstance name="2,8-PAD"/>
</packageinstances>
</package3d>
<package3d name="3,0-PAD" urn="urn:adsk.eagle:package:14280/1" type="box" library_version="2">
<description>MOUNTING PAD 3.0 mm, round</description>
<packageinstances>
<packageinstance name="3,0-PAD"/>
</packageinstances>
</package3d>
<package3d name="3,2-PAD" urn="urn:adsk.eagle:package:14282/1" type="box" library_version="2">
<description>MOUNTING PAD 3.2 mm, round</description>
<packageinstances>
<packageinstance name="3,2-PAD"/>
</packageinstances>
</package3d>
<package3d name="3,3-PAD" urn="urn:adsk.eagle:package:14283/1" type="box" library_version="2">
<description>MOUNTING PAD 3.3 mm, round</description>
<packageinstances>
<packageinstance name="3,3-PAD"/>
</packageinstances>
</package3d>
<package3d name="3,6-PAD" urn="urn:adsk.eagle:package:14284/1" type="box" library_version="2">
<description>MOUNTING PAD 3.6 mm, round</description>
<packageinstances>
<packageinstance name="3,6-PAD"/>
</packageinstances>
</package3d>
<package3d name="4,1-PAD" urn="urn:adsk.eagle:package:14285/1" type="box" library_version="2">
<description>MOUNTING PAD 4.1 mm, round</description>
<packageinstances>
<packageinstance name="4,1-PAD"/>
</packageinstances>
</package3d>
<package3d name="4,3-PAD" urn="urn:adsk.eagle:package:14286/1" type="box" library_version="2">
<description>MOUNTING PAD 4.3 mm, round</description>
<packageinstances>
<packageinstance name="4,3-PAD"/>
</packageinstances>
</package3d>
<package3d name="4,5-PAD" urn="urn:adsk.eagle:package:14287/1" type="box" library_version="2">
<description>MOUNTING PAD 4.5 mm, round</description>
<packageinstances>
<packageinstance name="4,5-PAD"/>
</packageinstances>
</package3d>
<package3d name="5,0-PAD" urn="urn:adsk.eagle:package:14288/1" type="box" library_version="2">
<description>MOUNTING PAD 5.0 mm, round</description>
<packageinstances>
<packageinstance name="5,0-PAD"/>
</packageinstances>
</package3d>
<package3d name="5,5-PAD" urn="urn:adsk.eagle:package:14291/1" type="box" library_version="2">
<description>MOUNTING PAD 5.5 mm, round</description>
<packageinstances>
<packageinstance name="5,5-PAD"/>
</packageinstances>
</package3d>
</packages3d>
<symbols>
<symbol name="MOUNT-PAD" urn="urn:adsk.eagle:symbol:14249/1" library_version="2">
<wire x1="0.254" y1="2.032" x2="2.032" y2="0.254" width="1.016" layer="94" curve="-75.749967" cap="flat"/>
<wire x1="-2.032" y1="0.254" x2="-0.254" y2="2.032" width="1.016" layer="94" curve="-75.749967" cap="flat"/>
<wire x1="-2.032" y1="-0.254" x2="-0.254" y2="-2.032" width="1.016" layer="94" curve="75.749967" cap="flat"/>
<wire x1="0.254" y1="-2.032" x2="2.032" y2="-0.254" width="1.016" layer="94" curve="75.749967" cap="flat"/>
<circle x="0" y="0" radius="1.524" width="0.0508" layer="94"/>
<text x="2.794" y="0.5842" size="1.778" layer="95">&gt;NAME</text>
<text x="2.794" y="-2.4638" size="1.778" layer="96">&gt;VALUE</text>
<pin name="MOUNT" x="-2.54" y="0" visible="off" length="short" direction="pas"/>
</symbol>
</symbols>
<devicesets>
<deviceset name="MOUNT-PAD-ROUND" urn="urn:adsk.eagle:component:14303/2" prefix="H" library_version="2">
<description>&lt;b&gt;MOUNTING PAD&lt;/b&gt;, round</description>
<gates>
<gate name="G$1" symbol="MOUNT-PAD" x="0" y="0"/>
</gates>
<devices>
<device name="2.8" package="2,8-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B2,8"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14281/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="6" constant="no"/>
</technology>
</technologies>
</device>
<device name="3.0" package="3,0-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B3,0"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14280/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="17" constant="no"/>
</technology>
</technologies>
</device>
<device name="3.2" package="3,2-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B3,2"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14282/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="4" constant="no"/>
</technology>
</technologies>
</device>
<device name="3.3" package="3,3-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B3,3"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14283/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="1" constant="no"/>
</technology>
</technologies>
</device>
<device name="3.6" package="3,6-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B3,6"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14284/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="0" constant="no"/>
</technology>
</technologies>
</device>
<device name="4.1" package="4,1-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B4,1"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14285/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="0" constant="no"/>
</technology>
</technologies>
</device>
<device name="4.3" package="4,3-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B4,3"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14286/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="0" constant="no"/>
</technology>
</technologies>
</device>
<device name="4.5" package="4,5-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B4,5"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14287/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="0" constant="no"/>
</technology>
</technologies>
</device>
<device name="5.0" package="5,0-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B5"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14288/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="0" constant="no"/>
</technology>
</technologies>
</device>
<device name="5.5" package="5,5-PAD">
<connects>
<connect gate="G$1" pin="MOUNT" pad="B5,5"/>
</connects>
<package3dinstances>
<package3dinstance package3d_urn="urn:adsk.eagle:package:14291/1"/>
</package3dinstances>
<technologies>
<technology name="">
<attribute name="POPULARITY" value="0" constant="no"/>
</technology>
</technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
<library name="ESP32-DEVKITV1">
<packages>
<package name="ESP32-DEVKITV1">
<pad name="1" x="-22.87" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="2" x="-20.33" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="3" x="-17.79" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="4" x="-15.25" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="5" x="-12.71" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="6" x="-10.17" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="7" x="-7.63" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="8" x="-5.09" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="9" x="-2.55" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="10" x="-0.01" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="11" x="2.53" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="12" x="5.07" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="13" x="7.61" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="14" x="10.15" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="15" x="12.69" y="-13.54" drill="1" diameter="1.9304"/>
<pad name="30" x="-22.87" y="11.23" drill="1" diameter="1.9304"/>
<pad name="29" x="-20.33" y="11.23" drill="1" diameter="1.9304"/>
<pad name="28" x="-17.79" y="11.23" drill="1" diameter="1.9304"/>
<pad name="27" x="-15.25" y="11.23" drill="1" diameter="1.9304"/>
<pad name="26" x="-12.71" y="11.23" drill="1" diameter="1.9304"/>
<pad name="25" x="-10.17" y="11.23" drill="1" diameter="1.9304"/>
<pad name="24" x="-7.63" y="11.23" drill="1" diameter="1.9304"/>
<pad name="23" x="-5.09" y="11.23" drill="1" diameter="1.9304"/>
<pad name="22" x="-2.55" y="11.23" drill="1" diameter="1.9304"/>
<pad name="21" x="-0.01" y="11.23" drill="1" diameter="1.9304"/>
<pad name="20" x="2.53" y="11.23" drill="1" diameter="1.9304"/>
<pad name="19" x="5.07" y="11.23" drill="1" diameter="1.9304"/>
<pad name="18" x="7.61" y="11.23" drill="1" diameter="1.9304"/>
<pad name="17" x="10.15" y="11.23" drill="1" diameter="1.9304"/>
<pad name="16" x="12.69" y="11.23" drill="1" diameter="1.9304"/>
<text x="-22.21" y="-11.2" size="1.016" layer="21" rot="R90">3V3</text>
<text x="-19.67" y="-11.2" size="1.016" layer="21" rot="R90">GND</text>
<text x="-17.13" y="-11.2" size="1.016" layer="21" rot="R90">IO15</text>
<text x="-14.59" y="-11.2" size="1.016" layer="21" rot="R90">IO2</text>
<text x="-12.05" y="-11.2" size="1.016" layer="21" rot="R90">IO4</text>
<text x="-9.51" y="-11.2" size="1.016" layer="21" rot="R90">IO16</text>
<text x="-6.97" y="-11.2" size="1.016" layer="21" rot="R90">IO17</text>
<text x="-4.43" y="-11.2" size="1.016" layer="21" rot="R90">IO5</text>
<text x="-1.89" y="-11.2" size="1.016" layer="21" rot="R90">IO18</text>
<text x="0.65" y="-11.2" size="1.016" layer="21" rot="R90">IO19</text>
<text x="3.19" y="-11.2" size="1.016" layer="21" rot="R90">IO21</text>
<text x="5.73" y="-11.2" size="1.016" layer="21" rot="R90">IO3</text>
<text x="8.27" y="-11.2" size="1.016" layer="21" rot="R90">IO1</text>
<text x="10.81" y="-11.2" size="1.016" layer="21" rot="R90">IO22</text>
<text x="13.35" y="-11.2" size="1.016" layer="21" rot="R90">IO23</text>
<text x="-22.19" y="6.52" size="1.016" layer="21" rot="R90">VIN</text>
<text x="-19.65" y="6.52" size="1.016" layer="21" rot="R90">GND</text>
<text x="-17.11" y="6.52" size="1.016" layer="21" rot="R90">IO13</text>
<text x="-14.57" y="6.52" size="1.016" layer="21" rot="R90">IO12</text>
<text x="-12.03" y="6.52" size="1.016" layer="21" rot="R90">IO14</text>
<text x="-9.49" y="6.52" size="1.016" layer="21" rot="R90">IO27</text>
<text x="-6.95" y="6.52" size="1.016" layer="21" rot="R90">IO26</text>
<text x="-4.41" y="6.52" size="1.016" layer="21" rot="R90">IO25</text>
<text x="-1.87" y="6.52" size="1.016" layer="21" rot="R90">IO33</text>
<text x="0.67" y="6.52" size="1.016" layer="21" rot="R90">IO32</text>
<text x="3.21" y="6.52" size="1.016" layer="21" rot="R90">IO35</text>
<text x="5.75" y="6.52" size="1.016" layer="21" rot="R90">IO34</text>
<text x="8.29" y="6.52" size="1.016" layer="21" rot="R90">VN</text>
<text x="10.83" y="6.52" size="1.016" layer="21" rot="R90">VP</text>
<text x="13.37" y="6.52" size="1.016" layer="21" rot="R90">EN</text>
<text x="-4.93" y="-3.18" size="1.9304" layer="21">ESP32-Devkit V1</text>
<wire x1="-33" y1="12.7" x2="19" y2="12.7" width="0.254" layer="21"/>
<wire x1="19" y1="12.7" x2="19" y2="-15.2" width="0.254" layer="21"/>
<wire x1="19" y1="-15.2" x2="-33" y2="-15.2" width="0.254" layer="21"/>
<wire x1="-33" y1="-15.2" x2="-33" y2="12.7" width="0.254" layer="21"/>
<text x="-24.13" y="13.97" size="1.27" layer="21">&gt;NAME</text>
<text x="5" y="-17.24" size="1.27" layer="27">ESP32-DEVKITV1</text>
</package>
</packages>
<symbols>
<symbol name="ESP32-DEVKITV1">
<wire x1="-25.4" y1="-12.7" x2="-25.4" y2="12.7" width="0.254" layer="94"/>
<wire x1="-25.4" y1="12.7" x2="16" y2="12.7" width="0.254" layer="94"/>
<wire x1="16" y1="12.7" x2="16" y2="-12.7" width="0.254" layer="94"/>
<wire x1="16" y1="-12.7" x2="-25.4" y2="-12.7" width="0.254" layer="94"/>
<pin name="3V3" x="-22.86" y="-17.78" length="middle" rot="R90"/>
<pin name="GND" x="-20.32" y="-17.78" length="middle" rot="R90"/>
<pin name="IO15" x="-17.78" y="-17.78" length="middle" rot="R90"/>
<pin name="IO2" x="-15.24" y="-17.78" length="middle" rot="R90"/>
<pin name="IO4" x="-12.7" y="-17.78" length="middle" rot="R90"/>
<pin name="IO16" x="-10.16" y="-17.78" length="middle" rot="R90"/>
<pin name="IO17" x="-7.62" y="-17.78" length="middle" rot="R90"/>
<pin name="IO5" x="-5.08" y="-17.78" length="middle" rot="R90"/>
<pin name="IO18" x="-2.54" y="-17.78" length="middle" rot="R90"/>
<pin name="IO19" x="0" y="-17.78" length="middle" rot="R90"/>
<pin name="IO21" x="2.54" y="-17.78" length="middle" rot="R90"/>
<pin name="IO3" x="5.08" y="-17.78" length="middle" rot="R90"/>
<pin name="IO1" x="7.62" y="-17.78" length="middle" rot="R90"/>
<pin name="IO22" x="10.16" y="-17.78" length="middle" rot="R90"/>
<pin name="IO23" x="12.7" y="-17.78" length="middle" rot="R90"/>
<pin name="EN" x="12.7" y="17.78" length="middle" rot="R270"/>
<pin name="VP" x="10.16" y="17.78" length="middle" rot="R270"/>
<pin name="VN" x="7.62" y="17.78" length="middle" rot="R270"/>
<pin name="IO34" x="5.08" y="17.78" length="middle" rot="R270"/>
<pin name="IO35" x="2.54" y="17.78" length="middle" rot="R270"/>
<pin name="IO32" x="0" y="17.78" length="middle" rot="R270"/>
<pin name="IO33" x="-2.54" y="17.78" length="middle" rot="R270"/>
<pin name="IO25" x="-5.08" y="17.78" length="middle" rot="R270"/>
<pin name="IO26" x="-7.62" y="17.78" length="middle" rot="R270"/>
<pin name="IO27" x="-10.16" y="17.78" length="middle" rot="R270"/>
<pin name="IO14" x="-12.7" y="17.78" length="middle" rot="R270"/>
<pin name="IO12" x="-15.24" y="17.78" length="middle" rot="R270"/>
<pin name="IO13" x="-17.78" y="17.78" length="middle" rot="R270"/>
<pin name="GND1" x="-20.32" y="17.78" length="middle" rot="R270"/>
<pin name="VIN" x="-22.86" y="17.78" length="middle" rot="R270"/>
<text x="-26.67" y="5.08" size="1.27" layer="95" rot="R90">&gt;NAME</text>
<text x="18.4" y="-12.7" size="1.27" layer="96" rot="R90">ESP32-DEVKITV1</text>
</symbol>
</symbols>
<devicesets>
<deviceset name="ESP32DEVKITV1">
<gates>
<gate name="G$1" symbol="ESP32-DEVKITV1" x="0" y="0"/>
</gates>
<devices>
<device name="" package="ESP32-DEVKITV1">
<connects>
<connect gate="G$1" pin="3V3" pad="1"/>
<connect gate="G$1" pin="EN" pad="16"/>
<connect gate="G$1" pin="GND" pad="2"/>
<connect gate="G$1" pin="GND1" pad="28"/>
<connect gate="G$1" pin="IO1" pad="13"/>
<connect gate="G$1" pin="IO12" pad="27"/>
<connect gate="G$1" pin="IO13" pad="28"/>
<connect gate="G$1" pin="IO14" pad="26"/>
<connect gate="G$1" pin="IO15" pad="3"/>
<connect gate="G$1" pin="IO16" pad="6"/>
<connect gate="G$1" pin="IO17" pad="7"/>
<connect gate="G$1" pin="IO18" pad="9"/>
<connect gate="G$1" pin="IO19" pad="10"/>
<connect gate="G$1" pin="IO2" pad="4"/>
<connect gate="G$1" pin="IO21" pad="11"/>
<connect gate="G$1" pin="IO22" pad="14"/>
<connect gate="G$1" pin="IO23" pad="15"/>
<connect gate="G$1" pin="IO25" pad="23"/>
<connect gate="G$1" pin="IO26" pad="24"/>
<connect gate="G$1" pin="IO27" pad="24"/>
<connect gate="G$1" pin="IO3" pad="12"/>
<connect gate="G$1" pin="IO32" pad="21"/>
<connect gate="G$1" pin="IO33" pad="22"/>
<connect gate="G$1" pin="IO34" pad="19"/>
<connect gate="G$1" pin="IO35" pad="20"/>
<connect gate="G$1" pin="IO4" pad="5"/>
<connect gate="G$1" pin="IO5" pad="8"/>
<connect gate="G$1" pin="VIN" pad="30"/>
<connect gate="G$1" pin="VN" pad="18"/>
<connect gate="G$1" pin="VP" pad="17"/>
</connects>
<technologies>
<technology name=""/>
</technologies>
</device>
</devices>
</deviceset>
</devicesets>
</library>
</libraries>
<attributes>
</attributes>
<variantdefs>
</variantdefs>
<classes>
<class number="0" name="default" width="0" drill="0">
</class>
</classes>
<parts>
<part name="U$1" library="Library" deviceset="PCA9685" device="PCA9685" package3d_urn="urn:adsk.eagle:package:38566682/2"/>
<part name="DC1" library="dc-dc-converter" library_urn="urn:adsk.eagle:library:208" deviceset="NME" device="" package3d_urn="urn:adsk.eagle:package:12313/1"/>
<part name="DC2" library="dc-dc-converter" library_urn="urn:adsk.eagle:library:208" deviceset="NME" device="" package3d_urn="urn:adsk.eagle:package:12313/1"/>
<part name="BATTERY+" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="BATTERY-" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="CAMERA2-" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="CAMERA2+" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="CAMERA1-" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="CAMERA1+" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="U$2" library="ESP32-DEVKITV1" deviceset="ESP32DEVKITV1" device=""/>
<part name="DC3" library="dc-dc-converter" library_urn="urn:adsk.eagle:library:208" deviceset="NME" device="" package3d_urn="urn:adsk.eagle:package:12313/1"/>
<part name="ENCODER_GND" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="ENCODER_A" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="ENCODER_B" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="ENCODER_VCC" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="PWM_ESC" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="GND_ESC" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="PWM_STEER" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="VCC_STEER" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
<part name="GND_STEER" library="holes" library_urn="urn:adsk.eagle:library:237" deviceset="MOUNT-PAD-ROUND" device="2.8" package3d_urn="urn:adsk.eagle:package:14281/1" value=""/>
</parts>
<sheets>
<sheet>
<plain>
<text x="50.8" y="81.28" size="1.778" layer="91">ESP32</text>
<text x="-48.26" y="-48.26" size="1.778" layer="91">STEP-UP 7.4-13V</text>
<text x="-73.66" y="-48.26" size="1.778" layer="91">STEP-UP 7.4-13V</text>
<text x="-139.7" y="-48.26" size="1.778" layer="91">STEP-DOWN 7.4-5V</text>
<text x="-86.36" y="154.94" size="1.778" layer="91">PCA9685</text>
<text x="187.96" y="152.4" size="1.778" layer="91">Camera 1</text>
<text x="121.92" y="149.86" size="1.778" layer="91">Camera 2</text>
<text x="78.74" y="-38.1" size="1.778" layer="91">Encoder</text>
<text x="-254" y="-10.16" size="1.778" layer="91">Battery 7.4V</text>
<text x="-132.08" y="220.98" size="1.778" layer="91">MOTOR</text>
<text x="-68.58" y="220.98" size="1.778" layer="91">SERVO STEER</text>
</plain>
<instances>
<instance part="U$1" gate="G$1" x="-96.52" y="114.3" smashed="yes"/>
<instance part="DC1" gate="G$1" x="-40.64" y="-12.7" smashed="yes" rot="R90">
<attribute name="NAME" x="-46.355" y="-22.86" size="1.778" layer="95" rot="R90"/>
<attribute name="VALUE" x="-30.48" y="-22.86" size="1.778" layer="96" rot="R90"/>
</instance>
<instance part="DC2" gate="G$1" x="-66.04" y="-12.7" smashed="yes" rot="R90">
<attribute name="NAME" x="-71.755" y="-22.86" size="1.778" layer="95" rot="R90"/>
<attribute name="VALUE" x="-55.88" y="-22.86" size="1.778" layer="96" rot="R90"/>
</instance>
<instance part="BATTERY+" gate="G$1" x="-231.14" y="-2.54" smashed="yes">
<attribute name="NAME" x="-228.346" y="-1.9558" size="1.778" layer="95"/>
<attribute name="VALUE" x="-228.346" y="-5.0038" size="1.778" layer="96"/>
</instance>
<instance part="BATTERY-" gate="G$1" x="-231.14" y="-17.78" smashed="yes">
<attribute name="NAME" x="-228.346" y="-17.1958" size="1.778" layer="95"/>
<attribute name="VALUE" x="-228.346" y="-20.2438" size="1.778" layer="96"/>
</instance>
<instance part="CAMERA2-" gate="G$1" x="114.3" y="139.7" smashed="yes">
<attribute name="NAME" x="117.094" y="140.2842" size="1.778" layer="95"/>
<attribute name="VALUE" x="117.094" y="137.2362" size="1.778" layer="96"/>
</instance>
<instance part="CAMERA2+" gate="G$1" x="139.7" y="139.7" smashed="yes">
<attribute name="NAME" x="142.494" y="140.2842" size="1.778" layer="95"/>
<attribute name="VALUE" x="142.494" y="137.2362" size="1.778" layer="96"/>
</instance>
<instance part="CAMERA1-" gate="G$1" x="175.26" y="139.7" smashed="yes">
<attribute name="NAME" x="178.054" y="140.2842" size="1.778" layer="95"/>
<attribute name="VALUE" x="178.054" y="137.2362" size="1.778" layer="96"/>
<attribute name="POPULARITY" x="175.26" y="139.7" size="1.778" layer="96" display="off"/>
</instance>
<instance part="CAMERA1+" gate="G$1" x="200.66" y="139.7" smashed="yes">
<attribute name="NAME" x="203.454" y="140.2842" size="1.778" layer="95"/>
<attribute name="VALUE" x="203.454" y="137.2362" size="1.778" layer="96"/>
</instance>
<instance part="U$2" gate="G$1" x="53.34" y="58.42" smashed="yes" rot="R90">
<attribute name="NAME" x="48.26" y="31.75" size="1.27" layer="95" rot="R180"/>
</instance>
<instance part="DC3" gate="G$1" x="-129.54" y="-12.7" smashed="yes" rot="R90">
<attribute name="NAME" x="-135.255" y="-22.86" size="1.778" layer="95" rot="R90"/>
<attribute name="VALUE" x="-119.38" y="-22.86" size="1.778" layer="96" rot="R90"/>
</instance>
<instance part="ENCODER_GND" gate="G$1" x="33.02" y="-27.94" smashed="yes">
<attribute name="NAME" x="35.814" y="-27.3558" size="1.778" layer="95"/>
<attribute name="VALUE" x="35.814" y="-30.4038" size="1.778" layer="96"/>
</instance>
<instance part="ENCODER_A" gate="G$1" x="60.96" y="-27.94" smashed="yes">
<attribute name="NAME" x="63.754" y="-27.3558" size="1.778" layer="95"/>
<attribute name="VALUE" x="63.754" y="-30.4038" size="1.778" layer="96"/>
</instance>
<instance part="ENCODER_B" gate="G$1" x="91.44" y="-27.94" smashed="yes">
<attribute name="NAME" x="94.234" y="-27.3558" size="1.778" layer="95"/>
<attribute name="VALUE" x="94.234" y="-30.4038" size="1.778" layer="96"/>
<attribute name="POPULARITY" x="91.44" y="-27.94" size="1.778" layer="96" display="off"/>
</instance>
<instance part="ENCODER_VCC" gate="G$1" x="116.84" y="-27.94" smashed="yes">
<attribute name="NAME" x="119.634" y="-27.3558" size="1.778" layer="95"/>
<attribute name="VALUE" x="119.634" y="-30.4038" size="1.778" layer="96"/>
</instance>
<instance part="PWM_ESC" gate="G$1" x="-142.24" y="210.82" smashed="yes">
<attribute name="NAME" x="-139.446" y="211.4042" size="1.778" layer="95"/>
<attribute name="VALUE" x="-139.446" y="208.3562" size="1.778" layer="96"/>
</instance>
<instance part="GND_ESC" gate="G$1" x="-116.84" y="210.82" smashed="yes">
<attribute name="NAME" x="-114.046" y="211.4042" size="1.778" layer="95"/>
<attribute name="VALUE" x="-114.046" y="208.3562" size="1.778" layer="96"/>
</instance>
<instance part="PWM_STEER" gate="G$1" x="-86.36" y="210.82" smashed="yes">
<attribute name="NAME" x="-83.566" y="211.4042" size="1.778" layer="95"/>
<attribute name="VALUE" x="-83.566" y="208.3562" size="1.778" layer="96"/>
</instance>
<instance part="VCC_STEER" gate="G$1" x="-63.5" y="210.82" smashed="yes">
<attribute name="NAME" x="-60.706" y="211.4042" size="1.778" layer="95"/>
<attribute name="VALUE" x="-60.706" y="208.3562" size="1.778" layer="96"/>
</instance>
<instance part="GND_STEER" gate="G$1" x="-38.1" y="210.82" smashed="yes">
<attribute name="NAME" x="-35.306" y="211.4042" size="1.778" layer="95"/>
<attribute name="VALUE" x="-35.306" y="208.3562" size="1.778" layer="96"/>
</instance>
</instances>
<busses>
</busses>
<nets>
<net name="N$1" class="0">
<segment>
<pinref part="U$1" gate="G$1" pin="GND02"/>
<wire x1="-12.7" y1="144.78" x2="86.36" y2="144.78" width="0.1524" layer="91"/>
<pinref part="U$2" gate="G$1" pin="GND"/>
<wire x1="86.36" y1="144.78" x2="86.36" y2="38.1" width="0.1524" layer="91"/>
<wire x1="86.36" y1="38.1" x2="71.12" y2="38.1" width="0.1524" layer="91"/>
<pinref part="ENCODER_GND" gate="G$1" pin="MOUNT"/>
<wire x1="30.48" y1="-27.94" x2="30.48" y2="-5.08" width="0.1524" layer="91"/>
<wire x1="30.48" y1="-5.08" x2="86.36" y2="-5.08" width="0.1524" layer="91"/>
<wire x1="86.36" y1="-5.08" x2="86.36" y2="38.1" width="0.1524" layer="91"/>
<junction x="86.36" y="38.1"/>
</segment>
</net>
<net name="N$2" class="0">
<segment>
<pinref part="U$1" gate="G$1" pin="VCC02"/>
<wire x1="-12.7" y1="124.46" x2="83.82" y2="124.46" width="0.1524" layer="91"/>
<pinref part="U$2" gate="G$1" pin="3V3"/>
<wire x1="83.82" y1="124.46" x2="83.82" y2="35.56" width="0.1524" layer="91"/>
<wire x1="83.82" y1="35.56" x2="71.12" y2="35.56" width="0.1524" layer="91"/>
<pinref part="ENCODER_VCC" gate="G$1" pin="MOUNT"/>
<wire x1="114.3" y1="-27.94" x2="114.3" y2="-10.16" width="0.1524" layer="91"/>
<wire x1="114.3" y1="-10.16" x2="83.82" y2="-10.16" width="0.1524" layer="91"/>
<wire x1="83.82" y1="-10.16" x2="83.82" y2="35.56" width="0.1524" layer="91"/>
<junction x="83.82" y="35.56"/>
</segment>
</net>
<net name="N$5" class="0">
<segment>
<pinref part="BATTERY-" gate="G$1" pin="MOUNT"/>
<wire x1="-233.68" y1="-17.78" x2="-160.02" y2="-17.78" width="0.1524" layer="91"/>
<wire x1="-160.02" y1="-17.78" x2="-160.02" y2="-43.18" width="0.1524" layer="91"/>
<pinref part="DC2" gate="G$1" pin="-VIN"/>
<wire x1="-160.02" y1="-43.18" x2="-127" y2="-43.18" width="0.1524" layer="91"/>
<wire x1="-127" y1="-43.18" x2="-63.5" y2="-43.18" width="0.1524" layer="91"/>
<wire x1="-63.5" y1="-43.18" x2="-63.5" y2="-25.4" width="0.1524" layer="91"/>
<pinref part="DC1" gate="G$1" pin="-VIN"/>
<wire x1="-63.5" y1="-43.18" x2="-38.1" y2="-43.18" width="0.1524" layer="91"/>
<wire x1="-38.1" y1="-43.18" x2="-38.1" y2="-25.4" width="0.1524" layer="91"/>
<junction x="-63.5" y="-43.18"/>
<pinref part="DC3" gate="G$1" pin="-VIN"/>
<wire x1="-127" y1="-25.4" x2="-127" y2="-43.18" width="0.1524" layer="91"/>
<junction x="-127" y="-43.18"/>
</segment>
</net>
<net name="N$6" class="0">
<segment>
<pinref part="BATTERY+" gate="G$1" pin="MOUNT"/>
<wire x1="-233.68" y1="-2.54" x2="-154.94" y2="-2.54" width="0.1524" layer="91"/>
<wire x1="-154.94" y1="-2.54" x2="-154.94" y2="-38.1" width="0.1524" layer="91"/>
<pinref part="DC2" gate="G$1" pin="+VIN"/>
<wire x1="-154.94" y1="-38.1" x2="-132.08" y2="-38.1" width="0.1524" layer="91"/>
<wire x1="-132.08" y1="-38.1" x2="-68.58" y2="-38.1" width="0.1524" layer="91"/>
<wire x1="-68.58" y1="-38.1" x2="-68.58" y2="-25.4" width="0.1524" layer="91"/>
<pinref part="DC1" gate="G$1" pin="+VIN"/>
<wire x1="-68.58" y1="-38.1" x2="-43.18" y2="-38.1" width="0.1524" layer="91"/>
<wire x1="-43.18" y1="-38.1" x2="-43.18" y2="-25.4" width="0.1524" layer="91"/>
<junction x="-68.58" y="-38.1"/>
<pinref part="DC3" gate="G$1" pin="+VIN"/>
<wire x1="-132.08" y1="-25.4" x2="-132.08" y2="-38.1" width="0.1524" layer="91"/>
<junction x="-132.08" y="-38.1"/>
</segment>
</net>
<net name="N$3" class="0">
<segment>
<pinref part="U$2" gate="G$1" pin="IO22"/>
<wire x1="71.12" y1="68.58" x2="78.74" y2="68.58" width="0.1524" layer="91"/>
<pinref part="U$1" gate="G$1" pin="SCL02"/>
<wire x1="78.74" y1="68.58" x2="78.74" y2="134.62" width="0.1524" layer="91"/>
<wire x1="78.74" y1="134.62" x2="-12.7" y2="134.62" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$4" class="0">
<segment>
<pinref part="U$1" gate="G$1" pin="SDA02"/>
<wire x1="-12.7" y1="129.54" x2="81.28" y2="129.54" width="0.1524" layer="91"/>
<pinref part="U$2" gate="G$1" pin="IO21"/>
<wire x1="81.28" y1="129.54" x2="81.28" y2="60.96" width="0.1524" layer="91"/>
<wire x1="81.28" y1="60.96" x2="71.12" y2="60.96" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$7" class="0">
<segment>
<pinref part="DC1" gate="G$1" pin="-VOUT"/>
<wire x1="-38.1" y1="0" x2="-38.1" y2="5.08" width="0.1524" layer="91"/>
<wire x1="-38.1" y1="5.08" x2="177.8" y2="5.08" width="0.1524" layer="91"/>
<pinref part="CAMERA1-" gate="G$1" pin="MOUNT"/>
<wire x1="177.8" y1="5.08" x2="177.8" y2="139.7" width="0.1524" layer="91"/>
<wire x1="177.8" y1="139.7" x2="172.72" y2="139.7" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$8" class="0">
<segment>
<pinref part="CAMERA1+" gate="G$1" pin="MOUNT"/>
<wire x1="198.12" y1="139.7" x2="198.12" y2="12.7" width="0.1524" layer="91"/>
<wire x1="198.12" y1="12.7" x2="-43.18" y2="12.7" width="0.1524" layer="91"/>
<pinref part="DC1" gate="G$1" pin="+VOUT"/>
<wire x1="-43.18" y1="12.7" x2="-43.18" y2="0" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$9" class="0">
<segment>
<pinref part="CAMERA2+" gate="G$1" pin="MOUNT"/>
<wire x1="137.16" y1="139.7" x2="137.16" y2="27.94" width="0.1524" layer="91"/>
<pinref part="DC2" gate="G$1" pin="+VOUT"/>
<wire x1="137.16" y1="27.94" x2="-68.58" y2="27.94" width="0.1524" layer="91"/>
<wire x1="-68.58" y1="27.94" x2="-68.58" y2="0" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$10" class="0">
<segment>
<pinref part="DC2" gate="G$1" pin="-VOUT"/>
<wire x1="-63.5" y1="0" x2="-63.5" y2="20.32" width="0.1524" layer="91"/>
<pinref part="CAMERA2-" gate="G$1" pin="MOUNT"/>
<wire x1="-63.5" y1="20.32" x2="111.76" y2="20.32" width="0.1524" layer="91"/>
<wire x1="111.76" y1="20.32" x2="111.76" y2="139.7" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$11" class="0">
<segment>
<pinref part="U$1" gate="G$1" pin="GND03"/>
<wire x1="-78.74" y1="116.84" x2="-78.74" y2="66.04" width="0.1524" layer="91"/>
<wire x1="-78.74" y1="66.04" x2="-127" y2="66.04" width="0.1524" layer="91"/>
<pinref part="DC3" gate="G$1" pin="-VOUT"/>
<wire x1="-127" y1="66.04" x2="-127" y2="0" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$12" class="0">
<segment>
<pinref part="DC3" gate="G$1" pin="+VOUT"/>
<wire x1="-132.08" y1="0" x2="-132.08" y2="71.12" width="0.1524" layer="91"/>
<wire x1="-132.08" y1="71.12" x2="-83.82" y2="71.12" width="0.1524" layer="91"/>
<pinref part="U$1" gate="G$1" pin="V+03"/>
<wire x1="-83.82" y1="71.12" x2="-83.82" y2="116.84" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$15" class="0">
<segment>
<pinref part="ENCODER_A" gate="G$1" pin="MOUNT"/>
<wire x1="58.42" y1="-27.94" x2="55.88" y2="-27.94" width="0.1524" layer="91"/>
<wire x1="55.88" y1="-27.94" x2="55.88" y2="-17.78" width="0.1524" layer="91"/>
<wire x1="55.88" y1="-17.78" x2="10.16" y2="-17.78" width="0.1524" layer="91"/>
<pinref part="U$2" gate="G$1" pin="IO34"/>
<wire x1="10.16" y1="-17.78" x2="10.16" y2="63.5" width="0.1524" layer="91"/>
<wire x1="10.16" y1="63.5" x2="35.56" y2="63.5" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$16" class="0">
<segment>
<pinref part="ENCODER_B" gate="G$1" pin="MOUNT"/>
<wire x1="88.9" y1="-27.94" x2="88.9" y2="-12.7" width="0.1524" layer="91"/>
<wire x1="88.9" y1="-12.7" x2="15.24" y2="-12.7" width="0.1524" layer="91"/>
<wire x1="15.24" y1="-12.7" x2="15.24" y2="60.96" width="0.1524" layer="91"/>
<pinref part="U$2" gate="G$1" pin="IO35"/>
<wire x1="15.24" y1="60.96" x2="35.56" y2="60.96" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$13" class="0">
<segment>
<wire x1="99.06" y1="147.32" x2="99.06" y2="134.62" width="0.1524" layer="91"/>
<wire x1="99.06" y1="134.62" x2="160.02" y2="134.62" width="0.1524" layer="91"/>
<wire x1="160.02" y1="134.62" x2="160.02" y2="147.32" width="0.1524" layer="91"/>
<wire x1="160.02" y1="147.32" x2="99.06" y2="147.32" width="0.1524" layer="91"/>
<wire x1="160.02" y1="147.32" x2="160.02" y2="134.62" width="0.1524" layer="91"/>
<wire x1="160.02" y1="134.62" x2="218.44" y2="134.62" width="0.1524" layer="91"/>
<wire x1="218.44" y1="134.62" x2="218.44" y2="137.16" width="0.1524" layer="91"/>
<wire x1="218.44" y1="137.16" x2="218.44" y2="147.32" width="0.1524" layer="91"/>
<wire x1="218.44" y1="147.32" x2="160.02" y2="147.32" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$14" class="0">
<segment>
<wire x1="12.7" y1="-22.86" x2="12.7" y2="-33.02" width="0.1524" layer="91"/>
<wire x1="12.7" y1="-33.02" x2="142.24" y2="-33.02" width="0.1524" layer="91"/>
<wire x1="142.24" y1="-33.02" x2="142.24" y2="-22.86" width="0.1524" layer="91"/>
<wire x1="142.24" y1="-22.86" x2="22.86" y2="-22.86" width="0.1524" layer="91"/>
<wire x1="22.86" y1="-22.86" x2="12.7" y2="-22.86" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$17" class="0">
<segment>
<wire x1="-236.22" y1="5.08" x2="-236.22" y2="-22.86" width="0.1524" layer="91"/>
<wire x1="-236.22" y1="-22.86" x2="-213.36" y2="-22.86" width="0.1524" layer="91"/>
<wire x1="-213.36" y1="-22.86" x2="-213.36" y2="-20.32" width="0.1524" layer="91"/>
<wire x1="-213.36" y1="-20.32" x2="-213.36" y2="5.08" width="0.1524" layer="91"/>
<wire x1="-213.36" y1="5.08" x2="-236.22" y2="5.08" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$19" class="0">
<segment>
<pinref part="U$1" gate="G$1" pin="GND$1"/>
<wire x1="-129.54" y1="134.62" x2="-134.62" y2="134.62" width="0.1524" layer="91"/>
<wire x1="-134.62" y1="134.62" x2="-134.62" y2="198.12" width="0.1524" layer="91"/>
<wire x1="-134.62" y1="198.12" x2="-119.38" y2="198.12" width="0.1524" layer="91"/>
<pinref part="GND_ESC" gate="G$1" pin="MOUNT"/>
<wire x1="-119.38" y1="198.12" x2="-119.38" y2="210.82" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$18" class="0">
<segment>
<pinref part="U$1" gate="G$1" pin="PWM$1"/>
<wire x1="-129.54" y1="144.78" x2="-132.08" y2="144.78" width="0.1524" layer="91"/>
<wire x1="-132.08" y1="144.78" x2="-132.08" y2="193.04" width="0.1524" layer="91"/>
<wire x1="-132.08" y1="193.04" x2="-144.78" y2="193.04" width="0.1524" layer="91"/>
<pinref part="PWM_ESC" gate="G$1" pin="MOUNT"/>
<wire x1="-144.78" y1="193.04" x2="-144.78" y2="210.82" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$20" class="0">
<segment>
<wire x1="-154.94" y1="215.9" x2="-154.94" y2="205.74" width="0.1524" layer="91"/>
<wire x1="-154.94" y1="205.74" x2="-101.6" y2="205.74" width="0.1524" layer="91"/>
<wire x1="-101.6" y1="205.74" x2="-101.6" y2="215.9" width="0.1524" layer="91"/>
<wire x1="-101.6" y1="215.9" x2="-154.94" y2="215.9" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$21" class="0">
<segment>
<wire x1="-99.06" y1="215.9" x2="-99.06" y2="205.74" width="0.1524" layer="91"/>
<wire x1="-99.06" y1="205.74" x2="-17.78" y2="205.74" width="0.1524" layer="91"/>
<wire x1="-17.78" y1="205.74" x2="-17.78" y2="215.9" width="0.1524" layer="91"/>
<wire x1="-17.78" y1="215.9" x2="-99.06" y2="215.9" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$22" class="0">
<segment>
<pinref part="PWM_STEER" gate="G$1" pin="MOUNT"/>
<wire x1="-88.9" y1="210.82" x2="-88.9" y2="193.04" width="0.1524" layer="91"/>
<wire x1="-88.9" y1="193.04" x2="-111.76" y2="193.04" width="0.1524" layer="91"/>
<pinref part="U$1" gate="G$1" pin="PWM$2"/>
<wire x1="-111.76" y1="193.04" x2="-111.76" y2="144.78" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$23" class="0">
<segment>
<pinref part="VCC_STEER" gate="G$1" pin="MOUNT"/>
<wire x1="-66.04" y1="210.82" x2="-66.04" y2="190.5" width="0.1524" layer="91"/>
<wire x1="-66.04" y1="190.5" x2="-114.3" y2="190.5" width="0.1524" layer="91"/>
<pinref part="U$1" gate="G$1" pin="VCC$2"/>
<wire x1="-114.3" y1="190.5" x2="-114.3" y2="139.7" width="0.1524" layer="91"/>
<wire x1="-114.3" y1="139.7" x2="-111.76" y2="139.7" width="0.1524" layer="91"/>
</segment>
</net>
<net name="N$24" class="0">
<segment>
<pinref part="GND_STEER" gate="G$1" pin="MOUNT"/>
<wire x1="-40.64" y1="210.82" x2="-40.64" y2="187.96" width="0.1524" layer="91"/>
<wire x1="-40.64" y1="187.96" x2="-116.84" y2="187.96" width="0.1524" layer="91"/>
<pinref part="U$1" gate="G$1" pin="GND$2"/>
<wire x1="-116.84" y1="187.96" x2="-116.84" y2="134.62" width="0.1524" layer="91"/>
<wire x1="-116.84" y1="134.62" x2="-111.76" y2="134.62" width="0.1524" layer="91"/>
</segment>
</net>
</nets>
</sheet>
</sheets>
</schematic>
</drawing>
<compatibility>
<note version="8.2" severity="warning">
Since Version 8.2, EAGLE supports online libraries. The ids
of those online libraries will not be understood (or retained)
with this version.
</note>
<note version="8.3" severity="warning">
Since Version 8.3, EAGLE supports URNs for individual library
assets (packages, symbols, and devices). The URNs of those assets
will not be understood (or retained) with this version.
</note>
<note version="8.3" severity="warning">
Since Version 8.3, EAGLE supports the association of 3D packages
with devices in libraries, schematics, and board files. Those 3D
packages will not be understood (or retained) with this version.
</note>
</compatibility>
</eagle>
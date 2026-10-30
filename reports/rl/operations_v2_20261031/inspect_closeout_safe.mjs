// Official read-only current-node/billing/snapshot proof. No connection fields.
import fs from 'node:fs';
import { MatpoolSession } from '/Users/yi/miniforge3/envs/matpool-machine-provider/share/matpool-machine-provider/node_modules/@matpool/matcli/dist/matpool-session.mjs';
const root='/Users/yi/Documents/projects/music/operations/numerical_calibration_20261031';
const rental=JSON.parse(fs.readFileSync(root+'/rental_state.json','utf8'));
if (rental.node_id!==2128889 || rental.phase!=='rented') throw new Error('Only this owned node');
const session=new MatpoolSession();
const unwrap=r=>r.data??r;
const [rawNode,rawBills,rawSnapshots]=await Promise.all([
  session.request('/node',{searchParams:{id:2128889}},{allowBareResponse:true}),
  session.request('/node/bill',{searchParams:{request_id:'ZW703Y'}},{allowBareResponse:true}),
  session.request('/snapshots',{searchParams:{resource_pool_id:rental.resource_pool_id,machine_category:0,status:3,page:1,per_page:100}},{allowBareResponse:true}),
]);
const user=unwrap(rawNode).userNode;
if(user?.node?.id!==2128889 || user.displayID!=='ZW703Y')throw new Error('Wrong node identity');
const status=user.status??user.node.status;
const released=[3,4,6].includes(status);
const bills=(unwrap(rawBills).bills??[]).map(b=>({
  subject:b.subject,start_time:b.startTime,end_time:b.endTime,
  start_seconds:b.timeInfo?.startTime?.seconds??null,end_seconds:b.timeInfo?.endTime?.seconds??null,
  amount_provider_cent:b.amount??null,
}));
function ended(b){
  if(Number(b.end_seconds)>0 && Number(b.end_seconds)>=Number(b.start_seconds??0))return true;
  return typeof b.end_time==='string' && !b.end_time.startsWith('0001') && Number.isFinite(Date.parse(b.end_time)) && Date.parse(b.end_time)>0;
}
const snapshots=(unwrap(rawSnapshots).snapshots??[]).filter(s=>[227273,229669].includes(Number(s.id)))
  .map(s=>({id:Number(s.id),name:s.flag?.subject??s.name,status:s.status,enabled:s.enabled,size_bytes:s.sizeByte??null}));
const value={utc:new Date().toISOString(),node_id:2128889,display_id:'ZW703Y',provider_status:status,node_released:released,
  bills,all_this_node_bills_ended:bills.length>0&&bills.every(ended),
  compute_billing_stopped_confirmed:released&&bills.length>0&&bills.every(ended),
  snapshots,unchanged_runtime_snapshot_completed:snapshots.some(s=>s.id===227273&&s.status===3&&s.enabled===true),
  credentials_or_connection_fields_recorded:false,read_only:true};
fs.writeFileSync(root+'/CLOSEOUT_PROVIDER_LATEST.json.writing',JSON.stringify(value,null,2)+'\n');
fs.renameSync(root+'/CLOSEOUT_PROVIDER_LATEST.json.writing',root+'/CLOSEOUT_PROVIDER_LATEST.json');
process.stdout.write(JSON.stringify(value,null,2)+'\n');
